from setuptools import setup, find_packages


setup(
    name="LeekAI",
    version="0.1.0",
    description="Pure Python CNN with CLI",
    author="Ahmad Shatnawi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "leekai=leekai.cli:main",
        ],
    },
)
