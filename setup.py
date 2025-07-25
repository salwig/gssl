# Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# install: pip install .
# develop: pip install --editable . // pip install -e .
import os
import toml
import pathlib
import sysconfig
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile
from setuptools import setup, find_packages


pyproject_text = pathlib.Path("pyproject.toml").read_text()
pyproject_data = toml.loads(pyproject_text)
build_type = pyproject_data["build-system"]["build-type"]


BUILD_TYPES = {
    "Release": ["-O3", "-DNDEBUG"],
    "Debug": ["-O0", "-g"],
    "RelWithDebInfo": ["-O2", "-g", "-DNDEBUG"],
    "MinSizeRel": ["-Os", "-DNDEBUG"],
}

include_dirs = [
    "GSSL/utils/cpp/include",
    "GSSL/algorithms/VGL/cpp/include",
    "GSSL/extern/vamm/vamm/cpp/include",
    "GSSL/extern/vamm/vamm/extern/eigen",
]
for lib in (
    "unordered",
    "assert",
    "container_hash",
    "config",
    "core",
    "predef",
    "throw_exception",
    "mp11",
    "describe",
    "static_assert",
):
    include_dirs += [
        f"GSSL/extern/vamm/vamm/extern/boost/{lib}/include"
    ]

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += [
    "-Wall",
    "-Wextra",
    # "-Wshadow",
    "-pedantic",
    "-Wno-unknown-pragmas",
    "-march=native",
]
extra_compile_args += BUILD_TYPES.get(build_type, [])

define_macros = [("PRECISION_T", "double"), ("EIGEN_DONT_PARALLELIZE", None)]

ext_modules = [
    Pybind11Extension(
        "cpp",
        [
            "GSSL/utils/cpp/src/Bindings.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ["-fopenmp"],
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
    Pybind11Extension(
        "cppVGL",
        [
            "GSSL/algorithms/VGL/cpp/src/Bindings.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ["-fopenmp"],
        extra_link_args=["-lgomp"],
        define_macros=define_macros,
        language="c++",
        cxx_std=17,
    ),
]

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
with ParallelCompile(default=0):
    setup(
        name="GSSL",
        version="0.1",
        packages=find_packages(),
        zip_safe=False,
        ext_modules=ext_modules,
        install_requires=[
            "torch",
            "numpy",
            "scikit-learn",
            "pandas",
            "scipy",
            f"emmi @ file://{PKG_DIR}/GSSL/extern/emmi",
            f"vamm @ file://{PKG_DIR}/GSSL/extern/vamm",
        ],
    )
