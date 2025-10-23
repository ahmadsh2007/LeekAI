import argparse
import os
import sys
import numpy as np
import json # For loading architecture def

# Use absolute imports for package context
try:
    from .model import CNNModel # Dynamic model
    from .train import train
    from .data_loader import load_data_from_directory, read_image
    from . import layers as layers_numpy # To set JIT flag
    from .compare import sanity_test # Import selftest
except ImportError:
    print("Error: Failed to import package modules.", file=sys.stderr)
    print("Please run this as a module: python -m <your_package_name> ...", file=sys.stderr)
    sys.exit(1)

# --- Default Hyperparameters ---
DEFAULT_LR = 0.01
DEFAULT_EPOCHS = 10
DEFAULT_ARCH = 'SimpleNet' # Default is still a predefined name
DEFAULT_BATCH = 32
# Include predefined names for easy access
ARCH_CHOICES = ['MicroNet', 'SimpleNet', 'LeNet-5']

# --------------------------- Subcommand handlers ---------------------------- #

def load_architecture_definition(filepath):
    """Loads architecture definition from a JSON file."""
    if not filepath or not os.path.exists(filepath):
        print(f"Error: Architecture definition file not found at '{filepath}'", file=sys.stderr)
        return None
    try:
        with open(filepath, 'r') as f:
            arch_def = json.load(f)
        if isinstance(arch_def, list) and all(isinstance(item, dict) for item in arch_def):
             print(f"Loaded custom architecture definition from: {filepath}")
             return arch_def
        else:
             print(f"Error: Invalid JSON format in '{filepath}'. Expected a list of layer dictionaries.", file=sys.stderr)
             return None
    except Exception as e:
        print(f"Error loading or parsing architecture file '{filepath}': {e}", file=sys.stderr)
        return None


def cmd_train(args: argparse.Namespace) -> None:
    # --- 1. Set JIT compilation ---
    if not args.jit:
        print("--- JIT COMPILATION DISABLED ---")
        layers_numpy.JIT_ENABLED = False

    # --- 2. Load data ---
    data_path = args.data
    # ... (data loading remains the same) ...
    if not os.path.exists(data_path):
        print(f"Error: training data path not found at '{data_path}'", file=sys.stderr)
        sys.exit(1)
    print(f"Loading data from {data_path}...")
    images, labels, class_names_list = load_data_from_directory(data_path) # Returns list
    if images is None:
        print("No images were found. Aborting.", file=sys.stderr)
        sys.exit(1)
    num_classes = len(class_names_list)
    print(f"Found {images.shape[0]} images across {num_classes} classes: {class_names_list}")


    # --- 3. Determine Architecture Definition ---
    arch_def_to_use = None
    arch_name_for_print = ""
    if args.arch_file:
        arch_def_to_use = load_architecture_definition(args.arch_file)
        if arch_def_to_use is None:
            sys.exit(1) # Error message printed in load function
        arch_name_for_print = f"Custom (from {os.path.basename(args.arch_file)})"
    elif args.arch in ARCH_CHOICES:
        arch_def_to_use = args.arch # Pass the predefined name string
        arch_name_for_print = args.arch
    else:
        # This case should be prevented by argparse choices, but good to have
        print(f"Error: Invalid architecture specified: {args.arch}", file=sys.stderr)
        print(f"Choose from predefined: {ARCH_CHOICES} or use --arch_file", file=sys.stderr)
        sys.exit(1)

    # --- 4. Initialize model ---
    try:
        model = CNNModel(architecture_def=arch_def_to_use, num_classes=num_classes)
        model.class_names = np.array(class_names_list, dtype=object) # Store class names
        print(f"Model initialized (arch={arch_name_for_print}) for {num_classes} classes.")
    except Exception as e:
         print(f"Error initializing model: {e}", file=sys.stderr)
         sys.exit(1)

    # --- 5. Train ---
    train(
        model,
        images,
        labels,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        clip_grad=args.clip,
        lr_decay=args.decay,
    )

    # --- 6. Save model (includes arch def now) ---
    if args.save:
        try:
            model.save(args.save)
        except Exception as e:
            print(f"Error saving model to {args.save}: {e}", file=sys.stderr)

def cmd_predict(args: argparse.Namespace) -> None:
    # --- 1. Set JIT compilation ---
    if not args.jit:
        print("--- JIT COMPILATION DISABLED ---")
        layers_numpy.JIT_ENABLED = False

    # --- 2. Load model (loads arch def AND params from npz) ---
    if not args.model or not os.path.exists(args.model):
         print(f"Error: Model parameters file (.npz) not found at '{args.model}'", file=sys.stderr)
         sys.exit(1)

    try:
        model = CNNModel.load(args.model)
    except Exception as e:
        print(f"Error loading model from {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Load image ---
    # ... (image loading remains the same) ...
    if not os.path.exists(args.image):
        print(f"Error: image not found at '{args.image}'", file=sys.stderr)
        sys.exit(1)
    img = read_image(args.image)
    if img is None:
        print(f"Error: Could not read image {args.image}", file=sys.stderr)
        sys.exit(1)
    img_batch = np.expand_dims(img, axis=0)

    # --- 4. Predict ---
    print(f"Running prediction using model '{model.architecture_name}'...")
    logits, _ = model.forward(img_batch)

    # Softmax for probabilities
    exp_z = np.exp(logits - np.max(logits))
    probs = (exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)).flatten()

    print("Probabilities:")

    class_labels = model.class_names # Loaded from model
    if class_labels is None or len(class_labels) == 0 or len(class_labels) != len(probs):
         print("Warning: Class names not found in model or mismatch length. Using defaults.")
         class_labels = [f"class_{i}" for i in range(len(probs))]

    results = list(zip(class_labels, probs))
    results.sort(key=lambda x: x[1], reverse=True)

    for cname, p in results:
        print(f"  {cname}: {p:.4f}")
    print(f"\nPredicted class: {results[0][0]}")

def cmd_selftest(args: argparse.Namespace) -> None:
    # ... (selftest remains the same) ...
    if not args.jit:
        print("--- JIT COMPILATION DISABLED ---")
        layers_numpy.JIT_ENABLED = False
    sanity_test()


# ------------------------------- CLI plumbing ------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dynamic NumPy/Numba CNN: Train/predict with predefined or custom architectures.",
    )
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        print("Running in GUI mode. Use 'python -m <packagename>' for CLI.")
        sys.exit(0)

    sub = p.add_subparsers(dest="cmd", required=True)

    # --- train parser ---
    pt = sub.add_parser("train", help="Train a model on an image folder hierarchy.")
    pt.add_argument( "--data", required=True, help="Path to training data directory.")

    # --- Architecture arguments ---
    arch_group = pt.add_mutually_exclusive_group(required=False) # Make optional, default handled later
    arch_group.add_argument(
        "--arch",
        choices=ARCH_CHOICES,
        # default=DEFAULT_ARCH, # Default handled manually based on arch_file
        help=f"Predefined model architecture (default: {DEFAULT_ARCH} if --arch_file not used).",
    )
    arch_group.add_argument(
        "--arch_file",
        default=None,
        help="Path to a JSON file defining a custom architecture.",
    )

    # --- Hyperparameters ---
    pt.add_argument( "--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Epochs (default: {DEFAULT_EPOCHS}).")
    pt.add_argument( "--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR}).")
    pt.add_argument( "--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH}).")
    pt.add_argument( "--clip", type=float, default=1.0, help="Gradient clip value (default: 1.0).")
    pt.add_argument( "--decay", type=float, default=0.95, help="LR decay per epoch (default: 0.95).")
    pt.add_argument( "--save", default="", help="Path to save trained model parameters AND architecture as .npz (optional).")
    pt.add_argument( '--no-jit', action='store_false', dest='jit', help="Disable Numba JIT.")
    pt.set_defaults(func=cmd_train, jit=True)


    # --- predict parser ---
    pp = sub.add_parser("predict", help="Run inference on a single image using a saved model.")
    pp.add_argument( "--model", required=True, help="Path to a saved .npz model file (contains params and arch def).")
    pp.add_argument("--image", required=True, help="Path to the image to classify.")
    pp.add_argument( '--no-jit', action='store_false', dest='jit', help="Disable Numba JIT.")
    pp.set_defaults(func=cmd_predict, jit=True)


    # --- selftest parser ---
    ps = sub.add_parser("selftest", help="Run small sanity test of core ops.")
    ps.add_argument( '--no-jit', action='store_false', dest='jit', help="Disable Numba JIT.")
    ps.set_defaults(func=cmd_selftest, jit=True)


    return p

def main() -> None:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        print("\nNote: To run the GUI, use: python -m <packagename> gui")
        sys.exit(1)

    args = parser.parse_args()

    # --- Manual default for train architecture ---
    if args.cmd == 'train' and not args.arch_file and not args.arch:
        print(f"No architecture specified, using default: {DEFAULT_ARCH}")
        args.arch = DEFAULT_ARCH
    elif args.cmd == 'train' and args.arch_file and args.arch:
        # This shouldn't happen due to mutually_exclusive_group, but belt-and-suspenders
        print("Error: Cannot use --arch and --arch_file together.", file=sys.stderr)
        sys.exit(1)


    args.func(args)

if __name__ == "__main__":
    main()