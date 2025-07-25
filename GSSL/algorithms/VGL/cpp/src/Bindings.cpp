/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#define CPPLIB_ENABLE_PYTHON_INTERFACE

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "VGL.h"

PYBIND11_MODULE(cppVGL, m) {
    /* See https://numpy.org/devdocs/user/basics.types.html */

    VGL::bind(m);
}