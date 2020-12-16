import glob
from setuptools import setup, find_packages

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

INCLUDE_DIRS = [
    r'C:\Users\arqui\Documents\cpp_libraries\eigen-3.3.8',
    r'C:\Users\arqui\Documents\cpp_libraries\armadillo-10.1.1\include',
    r'C:\Users\arqui\Documents\cpp_libraries\OpenBLAS.0.2.14.1\lib\native\include'
]

OPEN_BLAS_LIB = rf'C:\Users\arqui\Documents\cpp_libraries\OpenBLAS.0.2.14.1\lib\native\lib\x64'
EXTRA_OBJECTS = [
    *(glob.glob(f'{OPEN_BLAS_LIB}/*.a'))
]

ext_modules = [
    Pybind11Extension(
        "zonas_urbanas",
        sorted(["src/main.cpp"]),
        include_dirs=INCLUDE_DIRS,
        extra_objects=EXTRA_OBJECTS,
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="zonas_urbanas",
    version=__version__,
    author="Arquimedes Macedo",
    author_email="",
    url="",
    description="Modulo para realizar operacoes em matrizes na analise de grafos",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
