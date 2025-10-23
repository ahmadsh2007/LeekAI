import numpy as np
from numba import jit

# --- Global JIT Toggle ---
# The GUI can set this to False to disable JIT for comparison
JIT_ENABLED = True

def jit_decorator(func):
    """Conditionally applies the Numba JIT decorator."""
    if JIT_ENABLED:
        # nopython=True: No Python-mode fallback. Forces fast compilation.
        # cache=True: Caches the compiled function to disk.
        return jit(nopython=True, cache=True)(func)
    return func

# --- Initialization ---

def he_uniform_init(fan_in, fan_out, shape):
    """He uniform initialization for weights."""
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

def glorot_uniform_init(fan_in, fan_out, shape):
    """Glorot (Xavier) uniform initialization for weights."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

# --- Layer: Convolution ---

@jit_decorator
def convolve_forward(X, K, b, stride, padding):
    """
    Forward pass for Convolution.
    X: Input batch (N, C_in, H_in, W_in)
    K: Kernels (C_out, C_in, K_h, K_w)
    b: Biases (C_out,)
    """
    N, C_in, H_in, W_in = X.shape
    C_out, C_in_k, K_h, K_w = K.shape

    # Calculate output dimensions
    H_out = (H_in - K_h + 2 * padding) // stride + 1
    W_out = (W_in - K_w + 2 * padding) // stride + 1

    # Apply padding
    if padding > 0:
        # Create padded array manually for Numba compatibility
        X_padded = np.zeros((N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=X.dtype)
        X_padded[:, :, padding:padding+H_in, padding:padding+W_in] = X
    else:
        X_padded = X

    # Initialize output
    Z = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    # Naive convolution loop (prime candidate for JIT)
    for n in range(N):            # For each image in batch
        for c_out in range(C_out): # For each output channel
            for h_out in range(H_out):  # For each output row
                h_in_start = h_out * stride
                h_in_end = h_in_start + K_h
                for w_out in range(W_out): # For each output col
                    w_in_start = w_out * stride
                    w_in_end = w_in_start + K_w

                    # Get the 3D window from the padded input
                    window = X_padded[n, :, h_in_start:h_in_end, w_in_start:w_in_end]

                    # Convolve with the kernel for this output channel
                    # (element-wise multiply and sum)
                    Z[n, c_out, h_out, w_out] = np.sum(window * K[c_out]) + b[c_out]

    cache = (X, K, b, stride, padding)
    return Z, cache

@jit_decorator
def convolve_backward(dZ, cache):
    """
    Backward pass for Convolution.
    dZ: Gradient from next layer (N, C_out, H_out, W_out)
    cache: (X, K, b, stride, padding) from forward pass
    """
    X, K, b, stride, padding = cache
    N, C_in, H_in, W_in = X.shape
    C_out, C_in_k, K_h, K_w = K.shape
    N, C_out, H_out, W_out = dZ.shape

    # Ensure dZ is float32
    dZ = dZ.astype(np.float32)

    # Apply padding (if used in forward pass)
    if padding > 0:
        X_padded = np.zeros((N, C_in, H_in + 2 * padding, W_in + 2 * padding), dtype=X.dtype)
        X_padded[:, :, padding:padding+H_in, padding:padding+W_in] = X
    else:
        X_padded = X

    # Initialize gradients
    dX_padded = np.zeros_like(X_padded, dtype=X.dtype) # Grad for padded input
    dK = np.zeros_like(K, dtype=K.dtype)
    db = np.zeros_like(b, dtype=b.dtype)

    # Gradient for bias (sum dZ over N, H_out, W_out)
    for c_out in range(C_out):
        db[c_out] = np.sum(dZ[:, c_out, :, :])

    # Gradient for kernels (dK) and input (dX)
    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_out):
                h_in_start = h_out * stride
                h_in_end = h_in_start + K_h
                for w_out in range(W_out):
                    w_in_start = w_out * stride
                    w_in_end = w_in_start + K_w

                    # Get the gradient value
                    grad = dZ[n, c_out, h_out, w_out] # Already float32

                    # Get the window
                    window = X_padded[n, :, h_in_start:h_in_end, w_in_start:w_in_end] # Should be float32

                    # Accumulate dK (correlation)
                    dK[c_out] += window * grad

                    # Accumulate dX_padded (full convolution)
                    dX_padded[n, :, h_in_start:h_in_end, w_in_start:w_in_end] += K[c_out] * grad

    # Un-pad dX if necessary
    if padding > 0:
        dX = dX_padded[:, :, padding:padding+H_in, padding:padding+W_in]
    else:
        dX = dX_padded

    return dX, dK, db

# --- Layer: ReLU ---

@jit_decorator
def relu_forward(X):
    """Forward pass for ReLU."""
    Z = np.maximum(0, X)
    cache = X # Store original X
    return Z, cache

@jit_decorator
def relu_backward(dZ, cache):
    """Backward pass for ReLU."""
    X = cache
    # Ensure both dZ and the condition are float32
    dX = dZ.astype(np.float32) * (X > 0).astype(np.float32)
    return dX

# --- Layer: Max Pooling ---

@jit_decorator
def max_pool_forward(X, pool_h, pool_w, stride):
    """
    Forward pass for Max Pooling.
    X: Input (N, C, H_in, W_in)
    """
    N, C, H_in, W_in = X.shape
    H_out = (H_in - pool_h) // stride + 1
    W_out = (W_in - pool_w) // stride + 1

    Z = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # We must store the indices of the max values for backprop
    indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32) # Store (row, col)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                h_in_start = h_out * stride
                h_in_end = h_in_start + pool_h
                for w_out in range(W_out):
                    w_in_start = w_out * stride
                    w_in_end = w_in_start + pool_w

                    window = X[n, c, h_in_start:h_in_end, w_in_start:w_in_end]

                    # Find max value
                    Z[n, c, h_out, w_out] = np.max(window)

                    # --- Manually calculate 2D index ---
                    flat_idx = np.argmax(window)
                    h_idx = flat_idx // pool_w # Row index within window
                    w_idx = flat_idx % pool_w  # Column index within window

                    # Store the *absolute* index in the input X
                    indices[n, c, h_out, w_out, 0] = h_in_start + h_idx
                    indices[n, c, h_out, w_out, 1] = w_in_start + w_idx

    cache = (X, indices, pool_h, pool_w, stride)
    return Z, cache

@jit_decorator
def max_pool_backward(dZ, cache):
    """
    Backward pass for Max Pooling.
    dZ: Gradient from next layer (N, C, H_out, W_out)
    """
    X, indices, pool_h, pool_w, stride = cache
    N, C, H_in, W_in = X.shape

    # Ensure dZ is float32
    dZ = dZ.astype(np.float32)
    dX = np.zeros_like(X, dtype=X.dtype) # Will be float32 if X is

    for n in range(N):
        for c in range(C):
            for h_out in range(dZ.shape[2]):
                for w_out in range(dZ.shape[3]):
                    # Get the gradient
                    grad = dZ[n, c, h_out, w_out] # Already float32

                    # Get the absolute index where the max was found
                    h_abs = indices[n, c, h_out, w_out, 0]
                    w_abs = indices[n, c, h_out, w_out, 1]

                    # Add the gradient ONLY to that single location
                    dX[n, c, h_abs, w_abs] += grad

    return dX

# --- Layer: Flatten ---

@jit_decorator
def flatten_forward(X):
    """Forward pass for Flatten."""
    N, C, H, W = X.shape
    # Numba needs reshape result to be contiguous C-ordered
    Z = X.reshape(N, -1).copy() # Add .copy() for contiguity
    cache = (N, C, H, W) # Store original shape
    return Z, cache

@jit_decorator
def flatten_backward(dZ, cache):
    """Backward pass for Flatten."""
    N, C, H, W = cache
    # Ensure dZ is C-contiguous and float32 before reshape
    dZ_contig = np.ascontiguousarray(dZ).astype(np.float32)
    dX = dZ_contig.reshape(N, C, H, W)
    return dX

# --- Layer: Fully Connected (Dense) ---

@jit_decorator
def fc_forward(X, W, b):
    """
    Forward pass for Fully Connected.
    X: Input (N, D_in)
    W: Weights (D_in, D_out)
    b: Biases (D_out,)
    """
    # Ensure inputs are contiguous and float32 for Numba matmul optimization
    X_contig = np.ascontiguousarray(X).astype(np.float32)
    W_contig = np.ascontiguousarray(W).astype(np.float32)
    b_contig = np.ascontiguousarray(b).astype(np.float32)

    Z = X_contig @ W_contig + b_contig
    cache = (X_contig, W_contig, b_contig) # Cache float32 contiguous arrays
    return Z, cache

@jit_decorator
def fc_backward(dZ, cache):
    """
    Backward pass for Fully Connected.
    dZ: Gradient from next layer (N, D_out)
    """
    X, W, b = cache # X and W are already float32 contiguous from forward
    N = X.shape[0]

    # Ensure dZ is contiguous and float32
    dZ_contig = np.ascontiguousarray(dZ).astype(np.float32)

    # Numba optimized matrix multiplications - ensure operands are float32
    dX = dZ_contig @ W.T # W.T should inherit float32 type
    dW = X.T @ dZ_contig # X.T should inherit float32 type
    db = np.sum(dZ_contig, axis=0) # Sum preserves type

    return dX, dW, db

# --- Loss Function (Softmax + Cross-Entropy) ---

@jit_decorator
def softmax_loss(Z, y):
    """
    Computes Softmax probabilities and Cross-Entropy loss.
    Z: Logits (N, num_classes)
    y: True labels (N,) of type int
    """
    N, num_classes = Z.shape
    N_float = np.float32(N) # Cast N to float32 for division

    # Ensure Z is contiguous and float32
    Z_contig = np.ascontiguousarray(Z).astype(np.float32)

    # --- Replace np.max with explicit loop ---
    max_z = np.empty((N, 1), dtype=np.float32)
    for i in range(N):
        max_val = Z_contig[i, 0]
        for j in range(1, num_classes):
            if Z_contig[i, j] > max_val:
                max_val = Z_contig[i, j]
        max_z[i, 0] = max_val

    exp_z = np.exp(Z_contig - max_z) # Subtracting max_z (N, 1) uses broadcasting

    # --- Replace np.sum with explicit loop for keepdims ---
    sum_exp_z = np.empty((N, 1), dtype=np.float32)
    for i in range(N):
        row_sum = np.float32(0.0)
        for j in range(num_classes):
            row_sum += exp_z[i, j]
        sum_exp_z[i, 0] = row_sum

    # Add small epsilon to denominator to prevent division by zero
    epsilon_denom = np.float32(1e-12)
    probs = exp_z / (sum_exp_z + epsilon_denom) # Should remain float32

    # Cross-entropy loss
    epsilon_log = np.float32(1e-12)
    # Numba requires explicit loops for advanced indexing like probs[arange(N), y]
    log_probs = np.zeros(N, dtype=np.float32)
    for i in range(N):
        log_probs[i] = -np.log(probs[i, y[i]] + epsilon_log) # Add epsilon here too

    loss = np.sum(log_probs) / N_float # Divide by float N

    # Gradient of loss w.r.t Z
    dZ = probs.copy() # Make a copy to modify (is float32)
    # Numba requires explicit loops for this indexing too
    for i in range(N):
        dZ[i, y[i]] -= np.float32(1.0)
    dZ /= N_float # Divide by float N

    return loss, dZ, probs