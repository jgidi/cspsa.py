#!/usr/bin/env python3

from setuptools import setup

setup(
    name="cspsa",
    version="0.1.0",
    description="Complex Simultaneous Perturbation Stochastic Approximation",
    url="https://github.com/jgidi/cspsa.py",
    author="Jorge Gidi",
    author_email="jorgegidi@gmail.com",
    license="Apache 2.0",
    install_requires=["numpy", "scipy", "tqdm"],
)
