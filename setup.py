from setuptools import setup, find_packages

setup(
    name="xlm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "sentence_transformers",
        "pydantic",
        "numpy",
    ],
) 