"""Python library for interfacing with Motionblinds BLE motors."""

import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="motionblindsble",
    version="0.1.0",
    description=(
        "Python library for interfacing with Motionblinds"
        " using Bluetooth Low Energy (BLE)."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LennP/motionblindsble",
    author="LennP",
    author_email="lennperik@hotmail.nl",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "bleak",
        "bleak-retry-connector",
        "pycryptodome",
    ],
    tests_require=[],
    platforms=["any"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Home Automation",
    ],
)
