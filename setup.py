"""
tinyLMTune — Genetic-Algorithm-Optimised TinyBERT Fine-Tuning
"""

from setuptools import setup, find_packages

setup(
    name="tinylmtune",
    version="0.0.1",
    description="A lightweight Python library that automates TinyBERT fine-tuning with Genetic Algorithm hyperparameter optimisation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Manish Agrawal / Priyanka Chakraborty",
    author_email = "manishagrawal.datascience@gmail.com / priyanka08993@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "datasets==3.6.0",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "sentencepiece>=0.1.99",
        "matplotlib>=3.7"
    ],
    extras_require={
        "dev": ["pytest", "ruff", "mypy"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
