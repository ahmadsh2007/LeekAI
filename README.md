# LeekAI: Pure Python CNN with CLI

LeekAI is a tiny Convolutional Neural Network written entirely in pure Python (no NumPy, no deep learning frameworks). It includes a simple CLI for training, prediction, and testing, with optional Pillow support for real image loading.

---

## Features

* Pure Python CNN:

  * Convolution
  * ReLU activation
  * Max pooling
  * Fully-connected layers
  * Softmax
* Train on datasets structured as `./data/train/<class_name>/*.jpg`
* Save/load model weights as JSON
* CLI interface (`train`, `predict`, `selftest`)
* Optional Pillow integration (for real images) with fallback to a built-in simulator
* Gradient clipping & learning rate decay for stability

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/ahmadsh2007/LeekAI.git
cd LeekAI
pip install -e .
```

Optional extras:

```bash
pip install pillow kagglehub
```

---

## Usage

### Train

```bash
leekai train --data ./data/train --epochs 5 --lr 0.1 --pillow auto --save model.json
```

### Predict

```bash
leekai predict --model model.json --image ./test.jpg --pillow yes
```

### Self-test

```bash
leekai selftest
```

You can also run it as a module (because `__main__.py` is provided):

```bash
python -m leekai train --data ./data/train
```

---

## Project Structure

```
src/
└─ leekai/
   ├─ cli.py          # CLI entry point
   ├─ model.py        # CNN model (forward, backward, update, save/load)
   ├─ layers.py       # Core math ops (conv, relu, pooling, etc.)
   ├─ train.py        # Training loop (SGD, loss, accuracy)
   ├─ data_loader.py  # Image loader (Pillow or simulator)
   └─ compare.py      # Sanity tests for ops
```

---

## Dataset Credits

This project can be trained on any dataset organized as:

```
data/train/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
├── class2/
│   ├── img1.jpg
│   └── img2.jpg
```

For testing, we used the **Dog and Cat Classification Dataset** by **Bhavik Jikadara** on Kaggle.

Download with KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Path:", path)
```

Credits:

* Dataset author: Bhavik Jikadara (Kaggle)
* Platform: Kaggle

---

## Notes

* Intended for educational use and experimentation, not performance.
* Works without Pillow (deterministic simulator used instead).
* Developed with the help of ChatGPT and Google Gemini.

---

## License

MIT License