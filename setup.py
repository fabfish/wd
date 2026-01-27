
from setuptools import setup, find_packages

setup(
    name="wd_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "scipy",
        "seaborn"
    ],
)
