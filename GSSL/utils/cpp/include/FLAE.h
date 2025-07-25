/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <iostream>

#include "Numpy.h"

namespace FLAE {
// Fast Local Anchor Embedding (FLAE)
void FLAE(cRef<Matrix<>> X, cRef<Matrix<>> anchor, cRef<Matrix<long>> IDX, size_t knn, precision_t beta,
          Ref<Vector<>> data, Ref<Vector<long>> row, Ref<Vector<long>> col) {
    int n = X.rows();
    int d = X.cols();
    Matrix<> II = Matrix<>::Identity(knn, knn);
    II *= 1e-4;

#pragma omp parallel
    {
        // row-major storage: use the Upper triangular part for LLT
        Eigen::LLT<Matrix<>, Eigen::Upper> solver;
        Matrix<> z(knn, d);
        Matrix<> C(knn, knn);
        ColVector<> diagC(knn);
        ColVector<> ones(knn);
        ColVector<> w(knn);
        ones.fill(1.0);
#pragma omp for
        for (long i = 0; i < n; ++i) {
            z = anchor(IDX.row(i), Eigen::indexing::all).rowwise() - X.row(i);
            C.noalias() = z * z.transpose();
            diagC = C.diagonal();
            diagC = (diagC.array() / diagC.maxCoeff()).exp();
            diagC /= diagC.sum();
            C += beta * diagC.asDiagonal();
            C += II;

            solver.compute(C);
            w = solver.solve(ones);
            w /= w.sum();
            w = w.array().abs();
            w /= w.sum();

            // Assign results to data, row, and col to build spare matrix in python
            data.segment(i * knn, knn) = w;
            row.segment(i * knn, knn).fill(i);
            col.segment(i * knn, knn) = IDX.row(i);
        }
    }
}
#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void bind(py::module_& m) {
    m.def("FLAE", &FLAE, "X"_a.noconvert(), "anchor"_a.noconvert(), "IDX"_a.noconvert(), "knn"_a, "beta"_a,
          "data"_a.noconvert(), "row"_a.noconvert(), "col"_a.noconvert());
}
#endif
}  // namespace FLAE
