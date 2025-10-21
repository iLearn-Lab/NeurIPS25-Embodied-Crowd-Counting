from setuptools import setup, find_packages

setup(
    name="ecc",
    version="0.1.0",
    description="Embodied Crowd Counting for NeurIPS 2025",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.6.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.1.78",
        "pillow>=10.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
