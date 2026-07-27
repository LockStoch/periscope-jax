
import os
import io
import platform
from setuptools import setup
from setuptools.extension import Extension
import numpy as np

EXT_MODULES = []

if   ("linux" in platform.system().lower()):

    print("*Compiling for Linux")
    COMPILE_ARGS = [
        "-O3", "-flto", "-fopenmp", "-ffast-math", 
            "-fno-finite-math-only", "-march=native"]
    LINKER_ARGS = [
        "-O3", "-flto", "-fopenmp", "-ffast-math", 
            "-fno-finite-math-only", "-march=native"]

elif ("darwin" in platform.system().lower()):

    print("*Compiling for MacOS")
    COMPILE_ARGS = [
        "-O3", "-flto", "-fopenmp", "-ffast-math", 
            "-Xclang", "-fno-finite-math-only"]
    LINKER_ARGS = [
        "-O3", "-flto", "-lomp", "-ffast-math", 
            "-fno-finite-math-only"]

elif ("win" in platform.system.lower()):

    print("*Compiling for Windows")
    COMPILE_ARGS = [
        "/Ox", "/GS-", 
        "/GL", "/LTCG", "/openmp:llvm", "/fp:fast"]
    LINKER_ARGS = [
        "/Ox", "/GS-", 
        "/GL", "/LTCG", "/openmp:llvm", "/fp:fast"]
        
else:

    print("*Unknown operating system")

HERE = os.path.abspath(os.path.dirname(__file__))

try:
    from Cython.Build import cythonize
    import Cython.Compiler.Options
    
    Cython.Compiler.Options.annotate = True

    EXT_MODULES += cythonize(Extension(
        "_kx",
        sources=[os.path.join(HERE, "_kx.pyx")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()]),
        annotate=True
    )    
    EXT_MODULES += cythonize(Extension(
        "_kt",
        sources=[os.path.join(HERE, "_kt.pyx")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()]),
        annotate=True
    )
    EXT_MODULES += cythonize(Extension(
        "tde",
        sources=[os.path.join(HERE, "tde.pyx")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()]),
        annotate=True
    )
    EXT_MODULES += cythonize(Extension(
        "sal",
        sources=[os.path.join(HERE, "sal.pyx")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()]),
        annotate=True
    )

except ImportError:
    EXT_MODULES += [Extension(
        "_kx",
        sources=[os.path.join(HERE, "_kx.c")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()])
    ]
    EXT_MODULES += [Extension(
        "_kt",
        sources=[os.path.join(HERE, "_kt.c")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()])
    ]
    EXT_MODULES += [Extension(
        "tde",
        sources=[os.path.join(HERE, "tde.c")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()])
    ]
    EXT_MODULES += [Extension(
        "sal",
        sources=[os.path.join(HERE, "sal.c")],
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINKER_ARGS,
        include_dirs=[np.get_include()])
    ]

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
