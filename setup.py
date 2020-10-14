"""
Setup file for development/publishing.
"""
import os

from setuptools import find_packages, setup

req_path = "requirements.txt"
if os.path.exists(req_path):
    with open(req_path) as f:
        requirements = [r.strip() for r in f.readlines()]
else:
    requirements = []

setup(
    name="lightelligence-sdk-examples",
    version="0.0.0",
    author="Lightelligence",
    description=("SDK Examples code for running on benchmarks."),
    license="BSD",
    include_package_data=True,
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.6",
)
