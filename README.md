# LeekAI: Dynamic NumPy/Numba CNN with GUI

LeekAI is a Convolutional Neural Network written from scratch using **NumPy** for high-performance math and **Numba** for Just-in-Time (JIT) compilation. It includes a full-featured GUI with a dynamic architecture builder, plus a CLI for training, prediction, and testing.  
This project was rewritten from an original "pure Python" version to be a high-performance, usable tool for learning and experimenting with CNN architectures.

## Features
- **NumPy/Numba Backend:** Core operations accelerated for significant speed. 
- **Dynamic Architecture Builder:** A separate GUI window to visually define custom CNNs by adding, configuring, and reordering layers (Conv, ReLU, Pool, Flatten, Dense).  
- **Save/Load Custom Architectures:** Save your custom designs to `.json` files and load them back into the builder or the main application.
- **Predefined Models:** Includes `MicroNet` (Light), `SimpleNet` (Medium), and `LeekNet-5` (Heavy) as starting points.
- **Full GUI:** A standalone GUI (built with `tkinter`) to manage data loading, architecture selection (predefined or custom), training, parameter saving/loading (`.npz`), and prediction.
- **CLI Interface:** Powerful CLI for training (with predefined or custom `.json` architectures), prediction, and testing.
- **JIT Toggling:** Enable/disable Numba JIT compilation from the GUI or CLI.
- **Efficient Parameter Saving:** Trained model weights are saved using NumPy's `.npz` format. (Architecture is saved separately as `.json` or embedded in the `.npz` during training saves).

## Installation
1- Clone the repo (or use your local files).  
2- Install the project and its dependencies. It's recommended to do this in a virtual environment.
```
# Navigate to the project directory (where setup.py is)
cd /path/to/LeekAI

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Cmd/Git Bash:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip (good practice)
python -m pip install --upgrade pip

# Install the project in editable mode
pip install -e .
```
This installs `numpy`, `numba`, `pillow`, and `scipy`.
## Usage
1. **GUI Mode (Recommended)**  
Make sure your virtual environment is active ((venv) should be in your prompt).  
Run the `leekai` command with the `gui` argument:
```
leekai gui
```
Alternatively, you can run it as a module:
```
python -m leekai gui
```
### Workflow:
1- Load your training data (`Load Data`).  
2- Choose a predefined architecture OR click `Define Custom Arch...`, create your model in the builder, `Save Arch (.json)` (optional, but recommended), and click `Use This Architecture`.  
3- (Optional) Load a previously saved architecture (`Load Arch (.json)`).  
4- Configure training parameters (epochs, LR, etc.).  
5- Set a path for `Save Params (.npz)` if you want to save the trained weights.  
6- Click `Start Training`.  
7- After training (or loading trained params via `Load Params (.npz)`), select an image (`Image`: path) and click `Run Prediction`.


2. **CLI Mode**
Make sure your virtual environment is active.  
**Train with Predefined Architecture**  
```
leekai train --data ./path/to/train --arch LeNet-5 --epochs 15 --lr 0.005 --save trained_lenet5.npz
```
**Train with Custom Architecture**
```
# First, create 'my_arch.json' using the GUI builder and save it.
leekai train --data ./path/to/train --arch_file ./my_arch.json --epochs 20 --save trained_custom.npz
```

**Predict**  
```
# Use a model saved via GUI or CLI (it contains the necessary arch info)
leekai predict --model ./trained_custom.npz --image ./path/to/image.jpg
```
**Self-test**  
Run the built-in sanity test for the NumPy/Numba layers.
```
leekai selftest
```

# Project Structure
```
LeekAI/
├── src/
│   └── leekai/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── GUI.py
│       ├── architecture_builder.py
│       ├── model.py
│       ├── layers.py
│       ├── train.py
│       ├── data_loader.py
│       └── compare.py      # (Optional) Sanity tests for layers
│
├── README.md
├── setup.py 
├── pyproject.toml
├── requirements.txt
└── MANIFEST.in
```

# Dataset Credits
This project can be trained on any dataset organized as:
```
data/train/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
├── class2/
│   ├── img1.jpg
│   └── img2.jpg
...
```
For testing, we used the **Dog and Cat Classification Dataset** by **Bhavik Jikadara** on Kaggle.  
Download with KaggleHub (requires pip install kagglehub and Kaggle API credentials):
```
import kagglehub
# This downloads the dataset to a default location (~/.cache/kagglehub/...)
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Dataset downloaded to:", path)
# You will likely need to unzip/extract the data and potentially reorganize it
# into the 'data/train/cat/' and 'data/train/dog/' structure.
```

## Credits:
- Dataset author: Bhavik Jikadara (Kaggle)
- Platform: Kaggle

# Notes
- This is **educational code**. While significantly faster due to NumPy/Numba, accuracy depends heavily on architecture, data, and training time. It may not match frameworks like PyTorch/TensorFlow.
- **Pillow** is required for image loading.
- **Numba JIT** provides significant speedup but has an initial compilation overhead on the first run or after changes.
- Developed collaboratively, including assistance from AI tools like ChatGPT and Google Gemini during the transition phases.

# License
MIT License