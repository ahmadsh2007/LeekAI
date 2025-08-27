from .layers import convolve_2d, relu, max_pool_2d


def _print_mat(m):
    for r in m:
        print(" ".join(f"{v:6.2f}" for v in r))
    print()


def sanity_test():
    img = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
    k = [[[1, 0], [0, -1]]]

    print("Input:")
    _print_mat(img)

    print("Conv valid, stride=1:")
    feat = convolve_2d(img, k, stride=1, padding=0)
    _print_mat(feat)

    print("ReLU:")
    _print_mat(relu(feat))

    print("MaxPool 2x2:")
    _print_mat(max_pool_2d(feat, 2, 2))


if __name__ == "__main__":
    sanity_test()
