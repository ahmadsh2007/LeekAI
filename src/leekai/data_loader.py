import os
import math
import sys

IMG_WIDTH = 32
IMG_HEIGHT = 32
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")


# --------- Optional Pillow plumbing ------------------------------------------
def _maybe_get_pillow_reader(mode):
    """
    mode: 'auto' | 'yes' | 'no'
    returns a callable read_image(path) -> 2D float matrix in [0,1], or None if unavailable.
    """
    if mode == "no":
        return None
    if mode not in ("auto", "yes"):
        mode = "auto"

    try:
        from PIL import Image  # type: ignore
    except Exception:
        if mode == "yes":
            print("Pillow requested but not installed; falling back to simulator.")
        return None

    def read_with_pillow(file_path):
        with Image.open(file_path) as img:
            img = img.convert("L")  # grayscale
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            pix = list(img.getdata())
        mat = [pix[i : i + IMG_WIDTH] for i in range(0, len(pix), IMG_WIDTH)]
        scale = 1.0 / 255.0
        for r in range(len(mat)):
            row = mat[r]
            for c in range(len(row)):
                row[c] = row[c] * scale
        return mat

    return read_with_pillow


# --------- Built-in simulator (no third-party libs) ---------------------------
class PillowSimulator:
    """Deterministic fake image source; produces a 2D grayscale matrix."""

    def __init__(self, pixel_matrix):
        self.pixel_matrix = pixel_matrix
        self.height = len(pixel_matrix)
        self.width = len(pixel_matrix[0]) if self.height else 0

    @classmethod
    def open(cls, file_path):
        original_size = 100
        seed = sum(ord(ch) for ch in os.path.basename(file_path))
        M = [[[0, 0, 0] for _ in range(original_size)] for _ in range(original_size)]
        for r in range(original_size):
            for c in range(original_size):
                M[r][c][0] = (r * 3 + seed) % 256
                M[r][c][1] = (c * 5 + seed) % 256
                M[r][c][2] = ((r + c) * 7 + seed) % 256
        return cls(M)

    def convert(self, mode):
        if mode != "L":
            return self
        H, W = self.height, self.width
        gray = [[0 for _ in range(W)] for _ in range(H)]
        for r in range(H):
            row = self.pixel_matrix[r]
            grow = gray[r]
            for c in range(W):
                px = row[c]
                grow[c] = int(0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
        return PillowSimulator(gray)

    def resize(self, size):
        tw, th = size
        if self.width == 0 or self.height == 0:
            return self
        x_ratio = self.width / float(tw)
        y_ratio = self.height / float(th)
        out = [[0 for _ in range(tw)] for _ in range(th)]
        for r in range(th):
            sy = int(math.floor(r * y_ratio))
            if sy >= self.height:
                sy = self.height - 1
            grow = out[r]
            src_row = self.pixel_matrix[sy]
            for c in range(tw):
                sx = int(math.floor(c * x_ratio))
                if sx >= self.width:
                    sx = self.width - 1
                grow[c] = src_row[sx]
        return PillowSimulator(out)

    def getdata(self):
        flat = []
        for row in self.pixel_matrix:
            flat.extend(row)
        return flat


def _read_with_simulator(file_path):
    img = PillowSimulator.open(file_path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    pix = img.getdata()
    mat = [pix[i : i + IMG_WIDTH] for i in range(0, len(pix), IMG_WIDTH)]
    scale = 1.0 / 255.0
    for r in range(len(mat)):
        row = mat[r]
        for c in range(len(row)):
            row[c] = row[c] * scale
    return mat


# --------- Public API ---------------------------------------------------------
def read_image(file_path, pillow_mode="auto"):
    """
    Returns a 2D float matrix in [0,1]. Uses Pillow if available/allowed,
    otherwise falls back to the simulator.
    """
    reader = _maybe_get_pillow_reader(pillow_mode)
    if reader is not None:
        try:
            return reader(file_path)
        except Exception as e:
            print(
                f"Warning: Pillow failed on '{file_path}': {e}. Falling back to simulator."
            )
    # fallback
    return _read_with_simulator(file_path)


def load_data_from_directory(directory_path, pillow_mode="auto", quiet=False):
    images, labels = [], []
    class_names = sorted(
        [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]
    )
    class_to_label = {name: i for i, name in enumerate(class_names)}
    if not class_to_label:
        return [], [], []

    if not quiet:
        print(f"Found classes: {class_to_label}")

    for cname, lab in class_to_label.items():
        class_path = os.path.join(directory_path, cname)
        if not quiet:
            print(f"Loading images from: {class_path}")

        files = [
            f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)
        ]
        n = len(files)
        for i, fname in enumerate(files):
            if not quiet:
                progress = (i + 1) / max(1, n)
                bar_len = 40
                filled = int(bar_len * progress)
                bar = "█" * filled + "-" * (bar_len - filled)
                sys.stdout.write("\r" + f"|{bar}| {progress:.1%} Complete")
                sys.stdout.flush()

            path = os.path.join(class_path, fname)
            try:
                img = read_image(path, pillow_mode=pillow_mode)
                images.append(img)
                labels.append(lab)
            except Exception as e:
                print(f"\nWarning: Could not process '{fname}'. Skipping. Error: {e}")
        if not quiet:
            print()  # newline after progress bar
    return images, labels, class_names
