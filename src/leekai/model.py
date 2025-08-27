from .layers import (
    convolve_2d,
    relu,
    max_pool_2d,
    flatten,
    fully_connected,
    softmax,
    relu_backward,
    fully_connected_backward,
    initialize_kernel,
    initialize_weights,
)


class DeeperCNN:
    def __init__(self, num_classes=2):
        # fixed conv1 kernels
        sobel_x = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        sobel_y = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
        laplace = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
        identity = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        self.conv1_kernels = [sobel_x, sobel_y, laplace, identity]

        # learnable conv2 kernels
        self.conv2_kernels = initialize_kernel(size=3, depth=4, num_kernels=8)

        # FC params
        self.fc1_weights = None
        self.fc1_biases = None
        self.output_weights = None
        self.output_biases = None

        # grads
        self.grad_fc1_weights = None
        self.grad_fc1_biases = None
        self.grad_output_weights = None
        self.grad_output_biases = None

        self.num_classes = num_classes
        self.cache = {}

    def _ensure_fc_shapes(self, flattened_dim):
        if self.fc1_weights is None:
            hidden = 128
            self.fc1_weights = initialize_weights(flattened_dim, hidden)
            self.fc1_biases = [0.0] * hidden
            self.output_weights = initialize_weights(hidden, self.num_classes)
            self.output_biases = [0.0] * self.num_classes
            self.grad_fc1_weights = [[0.0] * len(row) for row in self.fc1_weights]
            self.grad_fc1_biases = [0.0] * len(self.fc1_biases)
            self.grad_output_weights = [[0.0] * len(row) for row in self.output_weights]
            self.grad_output_biases = [0.0] * len(self.output_biases)

    def forward(self, image_2d):
        self.cache.clear()
        self.cache["input_image"] = image_2d

        conv1 = [
            convolve_2d(image_2d, k, stride=1, padding=0) for k in self.conv1_kernels
        ]
        act1 = [relu(m) for m in conv1]
        pool1 = [max_pool_2d(m, pool_size=2, stride=2) for m in act1]

        conv2 = [convolve_2d(pool1, k, stride=1, padding=0) for k in self.conv2_kernels]
        act2 = [relu(m) for m in conv2]
        pool2 = [max_pool_2d(m, pool_size=2, stride=2) for m in act2]

        flat = flatten(pool2)
        self.cache["flatten"] = flat

        self._ensure_fc_shapes(len(flat))
        self.cache["fc1_in"] = flat
        z1 = fully_connected(flat, self.fc1_weights, self.fc1_biases)
        self.cache["fc1_out"] = z1
        a1 = relu(z1)

        self.cache["out_in"] = a1
        logits = fully_connected(a1, self.output_weights, self.output_biases)
        probs = softmax(logits)
        return probs

    def backward(self, prediction, actual_label):
        grad = prediction[:]
        grad[actual_label] -= 1.0
        dW_out, dB_out, d_in_out = fully_connected_backward(
            grad, self.cache["out_in"], self.output_weights
        )
        self.grad_output_weights = dW_out
        self.grad_output_biases = dB_out

        d_relu = relu_backward(d_in_out, self.cache["fc1_out"])

        dW_fc1, dB_fc1, _ = fully_connected_backward(
            d_relu, self.cache["fc1_in"], self.fc1_weights
        )
        self.grad_fc1_weights = dW_fc1
        self.grad_fc1_biases = dB_fc1

    def update(self, lr, clip=None):
        if clip is not None and clip > 0:
            c = clip
            for i in range(len(self.grad_fc1_weights)):
                row = self.grad_fc1_weights[i]
                for j in range(len(row)):
                    v = row[j]
                    if v > c:
                        v = c
                    elif v < -c:
                        v = -c
                    self.grad_fc1_weights[i][j] = v
            for i in range(len(self.grad_fc1_biases)):
                v = self.grad_fc1_biases[i]
                self.grad_fc1_biases[i] = c if v > c else (-c if v < -c else v)
            for i in range(len(self.grad_output_weights)):
                row = self.grad_output_weights[i]
                for j in range(len(row)):
                    v = row[j]
                    if v > c:
                        v = c
                    elif v < -c:
                        v = -c
                    self.grad_output_weights[i][j] = v
            for i in range(len(self.grad_output_biases)):
                v = self.grad_output_biases[i]
                self.grad_output_biases[i] = c if v > c else (-c if v < -c else v)

        for i in range(len(self.fc1_weights)):
            wrow = self.fc1_weights[i]
            grows = self.grad_fc1_weights[i]
            for j in range(len(wrow)):
                wrow[j] -= lr * grows[j]
            self.fc1_biases[i] -= lr * self.grad_fc1_biases[i]

        for i in range(len(self.output_weights)):
            wrow = self.output_weights[i]
            grows = self.grad_output_weights[i]
            for j in range(len(wrow)):
                wrow[j] -= lr * grows[j]
            self.output_biases[i] -= lr * self.grad_output_biases[i]

    # (De)serialization
    def to_dict(self):
        return {
            "num_classes": self.num_classes,
            "conv1_kernels": self.conv1_kernels,
            "conv2_kernels": self.conv2_kernels,
            "fc1_weights": self.fc1_weights,
            "fc1_biases": self.fc1_biases,
            "output_weights": self.output_weights,
            "output_biases": self.output_biases,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(num_classes=d.get("num_classes", 2))
        m.conv1_kernels = d["conv1_kernels"]
        m.conv2_kernels = d["conv2_kernels"]
        m.fc1_weights = d["fc1_weights"]
        m.fc1_biases = d["fc1_biases"]
        m.output_weights = d["output_weights"]
        m.output_biases = d["output_biases"]
        m.grad_fc1_weights = [[0.0] * len(row) for row in m.fc1_weights]
        m.grad_fc1_biases = [0.0] * len(m.fc1_biases)
        m.grad_output_weights = [[0.0] * len(row) for row in m.output_weights]
        m.grad_output_biases = [0.0] * len(m.output_biases)
        return m
