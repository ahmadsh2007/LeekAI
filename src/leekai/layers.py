import random
import math


def _he_uniform_limit(fan_in):
    return math.sqrt(6.0 / max(1, fan_in))


def initialize_kernel(size, depth, num_kernels):
    fan_in = depth * size * size
    a = _he_uniform_limit(fan_in)
    return [
        [
            [[random.uniform(-a, a) for _ in range(size)] for _ in range(size)]
            for _ in range(depth)
        ]
        for _ in range(num_kernels)
    ]


def initialize_weights(input_size, output_size):
    a = _he_uniform_limit(input_size)
    return [
        [random.uniform(-a, a) for _ in range(input_size)] for _ in range(output_size)
    ]


def _pad2d(mat, pad):
    if pad <= 0:
        return mat
    h = len(mat)
    w = len(mat[0]) if h else 0
    row0 = [0.0] * (w + 2 * pad)
    out = [row0[:] for _ in range(pad)]
    for r in range(h):
        out.append([0.0] * pad + mat[r][:] + [0.0] * pad)
    out.extend([row0[:] for _ in range(pad)])
    return out


def convolve_2d(input_, kernel_3d, stride=1, padding=0):
    if input_ and isinstance(input_[0][0], list):
        chans = input_
    else:
        chans = [input_]
    C = len(chans)
    Kh = len(kernel_3d[0])
    Kw = len(kernel_3d[0][0]) if Kh else 0
    Dc = min(C, len(kernel_3d))
    if padding > 0:
        chans = [_pad2d(ch, padding) for ch in chans]
    H = len(chans[0])
    W = len(chans[0][0]) if H else 0
    out_h = (H - Kh) // stride + 1
    out_w = (W - Kw) // stride + 1
    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    rng_ch = range(Dc)
    rng_kh = range(Kh)
    rng_kw = range(Kw)
    for oy in range(out_h):
        iy = oy * stride
        row_out = out[oy]
        for ox in range(out_w):
            ix = ox * stride
            s = 0.0
            for c in rng_ch:
                ch = chans[c]
                kch = kernel_3d[c]
                for ky in rng_kh:
                    r = ch[iy + ky]
                    wk = kch[ky]
                    base = ix
                    for kx in rng_kw:
                        s += r[base + kx] * wk[kx]
            row_out[ox] = s
    return out


def relu(data):
    if not data:
        return data
    if isinstance(data[0], list):
        return [[(v if v > 0.0 else 0.0) for v in row] for row in data]
    else:
        return [(v if v > 0.0 else 0.0) for v in data]


def max_pool_2d(matrix, pool_size=2, stride=2):
    h = len(matrix)
    w = len(matrix[0]) if h else 0
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    for oy in range(out_h):
        iy = oy * stride
        row_o = out[oy]
        for ox in range(out_w):
            ix = ox * stride
            m = -1e30
            for py in range(pool_size):
                r = matrix[iy + py]
                start = ix
                for px in range(pool_size):
                    v = r[start + px]
                    if v > m:
                        m = v
            row_o[ox] = m
    return out


def flatten(list_of_matrices):
    flat = []
    for m in list_of_matrices:
        for row in m:
            flat.extend(row)
    return flat


def fully_connected(input_vec, weights, biases):
    out = []
    for w_row, b in zip(weights, biases):
        s = 0.0
        for i in range(len(input_vec)):
            s += input_vec[i] * w_row[i]
        out.append(s + b)
    return out


def softmax(vec):
    if not vec:
        return []
    m = vec[0]
    for v in vec[1:]:
        if v > m:
            m = v
    exps = [math.exp(v - m) for v in vec]
    s = 0.0
    for e in exps:
        s += e
    inv = 1.0 / s if s != 0.0 else 0.0
    return [e * inv for e in exps]


def relu_backward(d_out, original_input):
    if isinstance(original_input[0], list):
        out = []
        for g_row, x_row in zip(d_out, original_input):
            out.append([(g if x > 0.0 else 0.0) for g, x in zip(g_row, x_row)])
        return out
    else:
        return [(g if x > 0.0 else 0.0) for g, x in zip(d_out, original_input)]


def fully_connected_backward(d_out, input_vec, weights):
    dW = []
    for g in d_out:
        dW.append([g * x for x in input_vec])
    dB = d_out[:]
    din = [0.0] * len(input_vec)
    for j in range(len(d_out)):
        gj = d_out[j]
        wj = weights[j]
        for i in range(len(input_vec)):
            din[i] += gj * wj[i]
    return dW, dB, din
