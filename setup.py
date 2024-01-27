"""Python library for interfacing with MotionBlinds using Bluetooth Low Energy (BLE)."""
import pathlib

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="motionblindsble",
    version="{{VERSION}}",
    description="Python library for interfacing with MotionBlinds using Bluetooth Low Energy (BLE).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LennP/PyPi-MotionBlinds_BLE",
    author="LennP",
    author_email="lennperik@hotmail.nl",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["bleak", "bleak-retry-connector", "pycryptodome", "pytz"],
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
