import os
import sys
import numpy as np
from PIL import Image

# Fixed image size
IMG_WIDTH = 32
IMG_HEIGHT = 32
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

def read_image(file_path):
    """
    Reads an image, converts to grayscale, resizes,
    and returns as a numpy array of shape (1, H, W)
    with values in [0, 1].
    """
    try:
        with Image.open(file_path) as img:
            img_gray = img.convert("L")
            img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            
            # Convert to numpy array, scale to [0, 1]
            img_arr = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Reshape to (C, H, W)
            return img_arr.reshape(1, IMG_HEIGHT, IMG_WIDTH)
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None

def load_data_from_directory(directory_path, quiet=False):
    """
    Loads all images from a directory, expecting subdirectories 
    for each class.
    Returns (images, labels, class_names) as numpy arrays.
    """
    images_list = []
    labels_list = []
    
    class_names = sorted(
        [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]
    )
    class_to_label = {name: i for i, name in enumerate(class_names)}
    
    if not class_to_label:
        if not quiet:
            print(f"Error: No class subdirectories found in {directory_path}", file=sys.stderr)
        return None, None, None

    if not quiet:
        print(f"Found classes: {class_to_label}")

    for cname, label in class_to_label.items():
        class_path = os.path.join(directory_path, cname)
        if not quiet:
            print(f"Loading images from: {class_path}")

        files = [
            f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)
        ]
        n = len(files)
        
        for i, fname in enumerate(files):
            if not quiet and (i % 20 == 0 or i == n-1):
                progress = (i + 1) / max(1, n)
                bar_len = 40
                filled = int(bar_len * progress)
                bar = "█" * filled + "-" * (bar_len - filled)
                sys.stdout.write("\r" + f"|{bar}| {progress:.1%} Complete")
                sys.stdout.flush()

            path = os.path.join(class_path, fname)
            img = read_image(path)
            if img is not None:
                images_list.append(img)
                labels_list.append(label)
        
        if not quiet:
            print()  # newline after progress bar

    if not images_list:
        if not quiet:
            print("Error: No valid images were loaded.", file=sys.stderr)
        return None, None, class_names

    # Convert lists to numpy arrays
    # Images: (N, 1, H, W)
    # Labels: (N,)
    all_images = np.stack(images_list, axis=0)
    all_labels = np.array(labels_list, dtype=np.int32)
    
    return all_images, all_labels, class_names