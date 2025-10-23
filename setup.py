from setuptools import setup, find_packages
import os # Needed to read requirements

# --- Function to read requirements ---
def read_requirements(filename="requirements.txt"):
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
            # Filter out comments and empty lines
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Installation might miss dependencies.")
        return []

setup(
    name="LeekAI",
    version="2.0.0", # Updated version
    description="Dynamic NumPy/Numba CNN with GUI and CLI", # Updated description
    author="Ahmad Shatnawi", # Keep your name
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    # --- Read dependencies from requirements.txt ---
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            # --- CORRECTED ENTRY POINT ---
            "leekai=leekai.__main__:main",
            # --- END CORRECTION ---
        ],
    },
     # Include README for PyPI
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # Add classifiers, URL etc. if planning to publish
    # url="YOUR_GITHUB_REPO_URL",
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
)

