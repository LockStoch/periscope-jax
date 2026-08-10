
import os
import io
import platform
from setuptools import setup
from setuptools.extension import Extension
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))

NAME = "PERISCOPE"
DESCRIPTION = "Geophysical fluid dynamics across scales"
AUTHOR = "Darren Engwirda"
AUTHOR_EMAIL = "d.engwirda@gmail.com"
URL = "https://github.com/dengwirda/periscope"
VERSION = "0.9.0"
REQUIRES_PYTHON = ">=3.6.0"
KEYWORDS = "Geophysical Fluid Dynamics Numerical Methods"

REQUIRED = [
    "cython", "numpy", "scipy", "xarray", "netCDF4"
]

CLASSIFY = [
    "Development Status :: 4 - Beta",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics"
]

try:
    with io.open(os.path.join(
            HERE, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()

except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="custom",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    keywords=KEYWORDS,
    url=URL,
    ext_modules=EXT_MODULES,
    install_requires=REQUIRED,
    classifiers=CLASSIFY
)
