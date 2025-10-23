import numpy as np
import time
import json # For saving/loading architecture def
from .layers import (
    he_uniform_init, glorot_uniform_init,
    convolve_forward, convolve_backward,
    relu_forward, relu_backward,
    max_pool_forward, max_pool_backward,
    flatten_forward, flatten_backward,
    fc_forward, fc_backward,
    softmax_loss
)

class CNNModel:
    """
    DYNAMIC CNN Model Class.
    Builds the model based on an architecture definition list.
    Handles parameters, forward pass, backward pass, saving, and loading.
    """
    def __init__(self, architecture_def, num_classes, input_shape=(1, 32, 32)):
        """
        Initializes the model dynamically based on the architecture definition.

        Args:
            architecture_def (list): A list of dictionaries, where each dict
                                     defines a layer (e.g., {'type': 'conv', 'filters': 8, ...}).
                                     Can also be a string name ('MicroNet', 'SimpleNet', 'LeNet-5')
                                     for predefined architectures.
            num_classes (int): Number of output classes.
            input_shape (tuple): Shape of the input image (C, H, W).
        """
        self.params = {}
        self.grads = {}
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.class_names = [f"class_{i}" for i in range(num_classes)] # Default

        # --- Handle predefined architectures for backward compatibility/ease ---
        if isinstance(architecture_def, str):
            self.architecture_name = architecture_def # Store the name
            self.architecture_def = self._get_predefined_arch_def(architecture_def)
            if not self.architecture_def:
                 raise ValueError(f"Unknown predefined architecture name: {architecture_def}")
        elif isinstance(architecture_def, list):
            self.architecture_name = "Custom" # Give it a generic name
            self.architecture_def = architecture_def
            # Ensure the last layer matches num_classes (important for custom models)
            if self.architecture_def[-1].get("type") == "dense":
                 self.architecture_def[-1]["units"] = num_classes
            else:
                 raise ValueError("The last layer in a custom architecture definition must be 'dense'.")
        else:
            raise TypeError("architecture_def must be a list of layer dicts or a predefined name string.")

        # --- Dynamically initialize parameters ---
        self._initialize_params_dynamic()

    def _get_predefined_arch_def(self, arch_name):
        """Returns the architecture definition list for predefined names."""
        if arch_name == 'MicroNet':
            # C1(6, 5x5) -> R1 -> P1 -> F1 -> FC1(64) -> R2 -> Out
            return [
                {"type": "conv", "filters": 6, "size": 5, "stride": 1, "padding": 0},
                {"type": "relu"},
                {"type": "pool", "size": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 64},
                {"type": "relu"},
                {"type": "dense", "units": self.num_classes} # Last layer units set dynamically
            ]
        elif arch_name == 'SimpleNet':
            # C1(8, 5x5) -> R1 -> P1 -> C2(16, 3x3) -> R2 -> P2 -> F1 -> FC1(128) -> R3 -> Out
            return [
                {"type": "conv", "filters": 8, "size": 5, "stride": 1, "padding": 0},
                {"type": "relu"},
                {"type": "pool", "size": 2, "stride": 2},
                {"type": "conv", "filters": 16, "size": 3, "stride": 1, "padding": 0},
                {"type": "relu"},
                {"type": "pool", "size": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 128},
                {"type": "relu"},
                {"type": "dense", "units": self.num_classes}
            ]
        elif arch_name == 'LeNet-5':
            # C1(6, 5x5) -> R1 -> P1 -> C2(16, 5x5) -> R2 -> P2 -> F1 -> FC1(120) -> R3 -> FC2(84) -> R4 -> Out
            return [
                {"type": "conv", "filters": 6, "size": 5, "stride": 1, "padding": 0},
                {"type": "relu"},
                {"type": "pool", "size": 2, "stride": 2},
                {"type": "conv", "filters": 16, "size": 5, "stride": 1, "padding": 0},
                {"type": "relu"},
                {"type": "pool", "size": 2, "stride": 2},
                {"type": "flatten"},
                {"type": "dense", "units": 120},
                {"type": "relu"},
                {"type": "dense", "units": 84},
                {"type": "relu"},
                {"type": "dense", "units": self.num_classes}
            ]
        else:
            return None # Unknown name


    def _initialize_params_dynamic(self):
        """Initializes parameters by looping through the architecture definition."""
        current_shape = self.input_shape # (C, H, W)
        conv_idx = 0
        dense_idx = 0

        for i, layer_def in enumerate(self.architecture_def):
            layer_type = layer_def.get("type")

            if layer_type == "conv":
                # Calculate input shape for this layer
                C_in, H_in, W_in = current_shape
                # Get params from definition
                filters = layer_def["filters"]
                k_size = layer_def["size"]
                stride = layer_def["stride"]
                padding = layer_def["padding"]

                # --- Parameter Keys (e.g., K0, b0, K1, b1, ...) ---
                k_key = f"K{conv_idx}"
                b_key = f"b{conv_idx}"

                # Calculate output shape (needed for fan_out approx)
                H_out = (H_in - k_size + 2 * padding) // stride + 1
                W_out = (W_in - k_size + 2 * padding) // stride + 1

                # Initialize parameters
                # Use Glorot for conv layers
                fan_in = C_in * k_size * k_size
                fan_out = filters * H_out * W_out # Approximate based on output size
                self.params[k_key] = glorot_uniform_init(fan_in, fan_out, (filters, C_in, k_size, k_size))
                self.params[b_key] = np.zeros(filters, dtype=np.float32)

                # Update current shape for the next layer
                current_shape = (filters, H_out, W_out)
                conv_idx += 1

            elif layer_type == "pool":
                # Calculate input shape
                C_in, H_in, W_in = current_shape
                # Get params
                pool_size = layer_def["size"]
                stride = layer_def["stride"]

                # Calculate output shape
                H_out = (H_in - pool_size) // stride + 1
                W_out = (W_in - pool_size) // stride + 1

                # Update current shape
                current_shape = (C_in, H_out, W_out)
                # No parameters for pooling

            elif layer_type == "flatten":
                # Calculate input shape (volume)
                C_in, H_in, W_in = current_shape
                flat_dim = C_in * H_in * W_in
                # Update current shape (now 1D)
                current_shape = (flat_dim,) # Tuple representing (Dim,)
                # No parameters for flatten

            elif layer_type == "dense":
                # Calculate input shape (must be 1D after flatten)
                if len(current_shape) != 1:
                    raise ValueError(f"Shape error at layer {i}: Dense layer must follow Flatten or another Dense layer. Got shape {current_shape}.")
                D_in = current_shape[0]
                # Get params
                units = layer_def["units"]

                # --- Parameter Keys (e.g., W0, b_W0, W1, b_W1, ...) ---
                w_key = f"W{dense_idx}"
                b_w_key = f"b_W{dense_idx}"

                # Initialize parameters
                # Use Glorot for dense layers too (can adjust if needed)
                self.params[w_key] = glorot_uniform_init(D_in, units, (D_in, units))
                self.params[b_w_key] = np.zeros(units, dtype=np.float32)

                # Update current shape for the next layer
                current_shape = (units,)
                dense_idx += 1

            elif layer_type == "relu":
                # No shape change, no parameters
                pass

            else:
                print(f"Warning: Skipping unknown layer type '{layer_type}' during parameter initialization.")

        # --- Initialize Gradients ---
        # Create zero arrays matching the shape of parameters
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}


    def forward(self, X):
        """Performs the forward pass dynamically based on architecture_def."""
        caches = {}
        current_A = X.astype(np.float32) # Start with input, ensure float32
        conv_idx = 0
        dense_idx = 0

        for i, layer_def in enumerate(self.architecture_def):
            layer_type = layer_def.get("type")
            # Create unique cache keys using layer index and type
            cache_key_base = f"L{i}_{layer_type}"

            if layer_type == "conv":
                k_key = f"K{conv_idx}"
                b_key = f"b{conv_idx}"
                stride = layer_def["stride"]
                padding = layer_def["padding"]
                # Z = Output before activation
                current_A, caches[cache_key_base] = convolve_forward(current_A, self.params[k_key], self.params[b_key], stride, padding)
                conv_idx += 1
            elif layer_type == "relu":
                 # A = Output after activation
                current_A, caches[cache_key_base] = relu_forward(current_A)
            elif layer_type == "pool":
                pool_size = layer_def["size"]
                stride = layer_def["stride"]
                # P = Output after pooling
                current_A, caches[cache_key_base] = max_pool_forward(current_A, pool_h=pool_size, pool_w=pool_size, stride=stride)
            elif layer_type == "flatten":
                # F = Output after flattening
                current_A, caches[cache_key_base] = flatten_forward(current_A)
            elif layer_type == "dense":
                w_key = f"W{dense_idx}"
                b_w_key = f"b_W{dense_idx}"
                 # Z = Output before activation (for dense layers before final)
                 # Or Z_out = Final output logits
                current_A, caches[cache_key_base] = fc_forward(current_A, self.params[w_key], self.params[b_w_key])
                dense_idx += 1
            else:
                print(f"Warning: Skipping unknown layer type '{layer_type}' during forward pass.")

        Z_out = current_A # The output of the last layer is the final logits
        return Z_out, caches

    def backward(self, dZ_out, caches):
        """Performs the backward pass dynamically."""
        current_dA = dZ_out # Start with gradient from softmax_loss
        conv_idx = len([ld for ld in self.architecture_def if ld.get('type') == 'conv']) - 1
        dense_idx = len([ld for ld in self.architecture_def if ld.get('type') == 'dense']) - 1

        # Iterate through layers in reverse order
        for i in range(len(self.architecture_def) - 1, -1, -1):
            layer_def = self.architecture_def[i]
            layer_type = layer_def.get("type")
            cache_key_base = f"L{i}_{layer_type}"
            # Retrieve the cache saved during the forward pass
            cache = caches[cache_key_base]

            if layer_type == "dense":
                w_key = f"W{dense_idx}"
                b_w_key = f"b_W{dense_idx}"
                # Calculate gradients w.r.t input (dA_prev), weights (dW), and biases (db)
                current_dA, dW, db = fc_backward(current_dA, cache)
                # Store parameter gradients
                self.grads[w_key] = dW
                self.grads[b_w_key] = db
                dense_idx -= 1
            elif layer_type == "flatten":
                # Reshape gradient back to volume shape
                current_dA = flatten_backward(current_dA, cache)
            elif layer_type == "pool":
                # Distribute gradient back through max locations
                current_dA = max_pool_backward(current_dA, cache)
            elif layer_type == "relu":
                 # Apply ReLU gradient mask
                current_dA = relu_backward(current_dA, cache)
            elif layer_type == "conv":
                k_key = f"K{conv_idx}"
                b_key = f"b{conv_idx}"
                # Calculate gradients w.r.t input (dA_prev), kernels (dK), and biases (db)
                current_dA, dK, db = convolve_backward(current_dA, cache)
                # Store parameter gradients
                self.grads[k_key] = dK
                self.grads[b_key] = db
                conv_idx -= 1
            else:
                 print(f"Warning: Skipping unknown layer type '{layer_type}' during backward pass.")

            # current_dA now holds the gradient to be passed to the *previous* layer (i-1)

    def update(self, lr, clip=None):
        """Updates model parameters using gradients."""
        for k in self.params:
            if k in self.grads:
                grad = self.grads[k]
                grad = grad.astype(self.params[k].dtype)
                if clip is not None:
                    np.clip(grad, -clip, clip, out=grad)
                self.params[k] -= lr * grad
            # No warning needed here, grads are initialized to zero anyway

    def save(self, path):
        """
        Saves model parameters AND architecture definition to an .npz file.
        Architecture is saved as a stringified JSON.
        """
        print(f"Saving model to {path}...")
        save_data = self.params.copy()

        # --- Store Architecture Definition ---
        # Convert list of dicts to JSON string
        arch_def_str = json.dumps(self.architecture_def)
        # Store as a NumPy array (savez needs arrays)
        save_data['__architecture_def_json__'] = np.array(arch_def_str)

        # --- Store Other Metadata ---
        save_data['__architecture_name__'] = np.array(self.architecture_name)
        save_data['__num_classes__'] = np.array(self.num_classes)
        save_data['__input_shape__'] = np.array(self.input_shape)
        save_data['__class_names__'] = np.array(self.class_names, dtype=object)

        np.savez(path, **save_data)
        print("Save complete.")

    @classmethod
    def load(cls, path):
        """
        Loads model parameters AND architecture definition from an .npz file.
        """
        print(f"Loading model from {path}...")
        data = np.load(path, allow_pickle=True)

        # --- Load Architecture Definition ---
        if '__architecture_def_json__' not in data:
            raise ValueError("Model file is missing essential architecture definition ('__architecture_def_json__'). Cannot load.")

        arch_def_str = str(data['__architecture_def_json__'])
        try:
            architecture_def = json.loads(arch_def_str)
            if not isinstance(architecture_def, list):
                raise ValueError("Architecture definition in file is not a valid list.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse architecture definition from model file: {e}")

        # --- Load Metadata ---
        num_classes = int(data['__num_classes__']) if '__num_classes__' in data else 0
        input_shape = tuple(data['__input_shape__']) if '__input_shape__' in data else (1, 32, 32)
        architecture_name = str(data['__architecture_name__']) if '__architecture_name__' in data else "CustomLoaded"

        if num_classes == 0:
            raise ValueError("Model file is missing __num_classes__ metadata.")

        # --- Create model instance using loaded definition ---
        # Use a temporary name, the actual definition is passed
        model = cls(architecture_def=architecture_def, num_classes=num_classes, input_shape=input_shape)
        model.architecture_name = architecture_name # Restore original name if available

        # --- Load Class Names ---
        if '__class_names__' in data:
            model.class_names = list(data['__class_names__'])
        else:
            print("Warning: Class names not found in model file. Using defaults.")
            model.class_names = [f"class_{i}" for i in range(num_classes)]

        # --- Load Parameters ---
        loaded_param_count = 0
        missing_params = []
        shape_mismatch = []

        for k in model.params: # Iterate through params expected by the loaded arch
            if k in data:
                if data[k].shape == model.params[k].shape:
                    model.params[k] = data[k].astype(np.float32)
                    loaded_param_count += 1
                else:
                    shape_mismatch.append(f"'{k}' (Expected {model.params[k].shape}, got {data[k].shape})")
            else:
                missing_params.append(f"'{k}'")

        print(f"Loaded model (arch={model.architecture_name}, classes={num_classes})")
        if missing_params:
            print(f"Warning: Parameters not found in file (using initialized): {', '.join(missing_params)}")
        if shape_mismatch:
             print(f"Warning: Parameters with shape mismatch (using initialized): {', '.join(shape_mismatch)}")
        if loaded_param_count == 0 and (missing_params or shape_mismatch):
             print("Critical Warning: No parameters were successfully loaded. Model will use random initialization.")


        # Re-initialize gradients (important!)
        model.grads = {k: np.zeros_like(v) for k, v in model.params.items()}

        return model