from setuptools import setup, find_packages

setup(
    name="TitokTokenizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "einops",
        "torchvision",
        "tqdm",
        "Pillow",
    ],
) 