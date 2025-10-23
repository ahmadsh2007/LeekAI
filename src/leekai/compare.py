import numpy as np
from . import layers as np_layers

def sanity_test():
    """
    Runs a simple selftest of the core NumPy/Numba layers.
    """
    print("--- Running NumPy/Numba Sanity Test ---")
    
    # 1. Test Image (N=1, C=1, H=4, W=4)
    img = np.array([[[
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ]]], dtype=np.float32)
    
    # 2. Test Kernel (C_out=1, C_in=1, K_h=2, K_w=2)
    k = np.array([[[
        [1, 0],
        [0, -1]
    ]]], dtype=np.float32)
    
    # 3. Test Bias (C_out=1)
    b = np.array([0.0], dtype=np.float32)

    print("Input Image (1x1x4x4):\n", img[0, 0])
    print("\nKernel (1x1x2x2):\n", k[0, 0])

    # Test Conv
    print("\n--- Testing Convolve (stride=1, pad=0) ---")
    conv_out, conv_cache = np_layers.convolve_forward(img, k, b, stride=1, padding=0)
    print("Output (1x1x3x3):\n", conv_out[0, 0])
    
    # Expected: (e.g., 1*1 + 2*0 + 2*0 + 3*(-1) = -2)
    # [[-2, -2, -2],
    #  [-2, -2, -2],
    #  [-2, -2, -2]]
    assert np.all(conv_out[0, 0] == -2.0)
    print("Conv test PASSED.")

    # Test ReLU
    print("\n--- Testing ReLU ---")
    relu_out, relu_cache = np_layers.relu_forward(conv_out)
    print("Output:\n", relu_out[0, 0])
    # Expected: all 0s
    assert np.all(relu_out[0, 0] == 0.0)
    print("ReLU test PASSED.")

    # Test MaxPool
    print("\n--- Testing MaxPool (2x2, stride=1) on Conv Output ---")
    pool_out, pool_cache = np_layers.max_pool_forward(conv_out, pool_h=2, pool_w=2, stride=1)
    print("Output (1x1x2x2):\n", pool_out[0, 0])
    # Expected: [[-2, -2], [-2, -2]]
    assert np.all(pool_out[0, 0] == -2.0)
    print("MaxPool test PASSED.")
    
    print("\n--- Sanity Test Complete. All tests passed. ---")

if __name__ == "__main__":
    # This allows running `python -m leekai.compare`
    sanity_test()