# main.py
# CLI entry point: train / predict / selftest
#
# Examples:
#   python main.py train --data ./data/train --epochs 5 --lr 0.1 --pillow auto --save model.json
#   python main.py predict --model model.json --image ./some.jpg --pillow yes
#   python main.py selftest

import argparse
import json
import os
import sys

from model import DeeperCNN
from train import train
from data_loader import load_data_from_directory, read_image

DEFAULT_LR = 0.10
DEFAULT_EPOCHS = 10


def _save_model(model, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f)
    print(f"Saved model to: {path}")


def _load_model(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"Loaded model from: {path}")
    return DeeperCNN.from_dict(d)


def cmd_train(args):
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Error: Training data path not found at '{data_path}'")
        print("Expected structure: <data>/<class_name>/*.jpg")
        sys.exit(1)

    print(f"Loading data from {data_path} (pillow={args.pillow})...")
    images, labels, class_names = load_data_from_directory(
        data_path, pillow_mode=args.pillow
    )

    if not images:
        print("No images were found in the training directory. Aborting.")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"Found {len(images)} images across {num_classes} classes: {class_names}")

    model = DeeperCNN(num_classes=num_classes)
    print(f"Model initialized for {num_classes} classes.")

    train(
        model,
        images,
        labels,
        epochs=args.epochs,
        learning_rate=args.lr,
        clip_grad=args.clip,
        lr_decay=args.decay,
    )

    if args.save:
        _save_model(model, args.save)


def cmd_predict(args):
    if args.model and os.path.exists(args.model):
        model = _load_model(args.model)
    else:
        print(
            "No model file provided (or path not found). Initializing a fresh (untrained) model with 2 classes."
        )
        model = DeeperCNN(num_classes=2)

    if not os.path.exists(args.image):
        print(f"Error: image not found at '{args.image}'")
        sys.exit(1)

    img = read_image(args.image, pillow_mode=args.pillow)
    probs = model.forward(img)
    if not probs:
        print("Failed to compute prediction.")
        sys.exit(1)

    # Pretty print
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  class_{i}: {p:.4f}")
    print(f"Predicted class: class_{probs.index(max(probs))}")


def cmd_selftest(_args):
    # lightweight import to avoid circulars
    from compare import sanity_test

    sanity_test()


def build_parser():
    p = argparse.ArgumentParser(
        description="Tiny CNN (pure Python) CLI with optional Pillow image loading."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train a model on an image folder hierarchy.")
    pt.add_argument(
        "--data",
        required=True,
        help="Path to training data directory (subfolders are class names).",
    )
    pt.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default {DEFAULT_EPOCHS}).",
    )
    pt.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default {DEFAULT_LR}).",
    )
    pt.add_argument(
        "--clip", type=float, default=1.0, help="Gradient clip value (default 1.0)."
    )
    pt.add_argument(
        "--decay", type=float, default=0.95, help="LR decay per epoch (default 0.95)."
    )
    pt.add_argument(
        "--pillow",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Use Pillow if available (default auto).",
    )
    pt.add_argument(
        "--save", default="", help="Path to save trained model as JSON (optional)."
    )
    pt.set_defaults(func=cmd_train)

    # predict
    pp = sub.add_parser("predict", help="Run inference on a single image.")
    pp.add_argument(
        "--model",
        required=False,
        default="",
        help="Path to a saved model.json (optional).",
    )
    pp.add_argument("--image", required=True, help="Path to the image to classify.")
    pp.add_argument(
        "--pillow",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Use Pillow if available (default auto).",
    )
    pp.set_defaults(func=cmd_predict)

    # selftest
    ps = sub.add_parser("selftest", help="Run small sanity test of core ops.")
    ps.set_defaults(func=cmd_selftest)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
