/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <iostream>

#include "Numpy.h"

namespace LAE {
// Function to project a vector onto the simplex
void SimplexPr(cRef<Vector<>> v, Ref<Vector<>> u, Ref<Vector<>> z) {
    u = v;
    std::sort(u.begin(), u.end(), std::greater<precision_t>());
    precision_t sum = 0.0;
    precision_t theta = 0.0;
    precision_t theta_rho = 0.0;
    for (size_t i = 0; i < (size_t)u.size(); ++i) {
        sum += u[i];
        theta = (sum - 1.0) / (i + 1);
        if (u[i] - theta > 0) {
            theta_rho = theta;
        }
    }
    z = v.array() - theta_rho;
    z = z.cwiseMax(0);
}

// Local Anchor Embedding (LAE)
void LAE(cRef<Matrix<>> X, cRef<Matrix<>> Anchor, cRef<Matrix<long>> pos,
         Ref<Matrix<>> val, size_t cn) {
    const size_t n = X.rows(); // old: X.cols()
    const size_t d = X.cols(); // old: X.rows()
    const size_t s = pos.cols();

#pragma omp parallel
    {
        Matrix<> U(d, s);
        ColVector<> x(d);

        ColVector<> z1(s);
        ColVector<> z0(s);
        ColVector<> dgv(s);
        ColVector<> v(s);
        ColVector<> u(s);
        ColVector<> z(s);

        ColVector<> dif(d);

        ColVector<> delta(cn + 2);
        ColVector<> beta(cn + 1);

        precision_t alpha;
        precision_t gv;
        precision_t b;
        precision_t gz;
        precision_t gvz;

#pragma omp for
        for (size_t i = 0; i < n; ++i) {
            x = X.row(i); // old: x = X.col(i);
            U = Anchor(pos.row(i), Eigen::indexing::all).transpose(); // old: U = Anchor(Eigen::indexing::all, pos.row(i));

            x /= x.norm();
            U = (U.array().rowwise() * U.array().square().colwise().sum().rsqrt()).matrix();

            // LAE
            z0.fill(1.0 / s);
            z1 = z0;
            delta.fill(0.0);
            delta(1) = 1;
            beta.fill(0.0);
            beta(0) = 1;

            for (size_t t = 0; t < cn; ++t) {
                alpha = (delta(t) - 1) / delta(t + 1);
                v = z1 + alpha * (z1 - z0);

                // dif = x - U * v;
                dif = x;
                dif.noalias() -= U * v;

                gv = 0.5 * dif.squaredNorm();
                // dgv = U.transpose() * (U * v - x);
                dgv.noalias() = U.transpose() * (-dif);  // reused dif here

                for (size_t j = 0; j < 101; ++j) {
                    // b = std::pow(2.0, j) * beta(t);
                    b = (1 << j) * beta(t);  // (1 << j) = 2**j
                    SimplexPr(v - dgv / b, u, z);
                    // dif = x - U * z;
                    dif = x;
                    dif.noalias() -= U * z;
                    gz = 0.5 * dif.dot(dif);
                    dif = z - v;
                    gvz = gv + dgv.dot(dif) + 0.5 * b * dif.dot(dif);  // dif.squaredNorm()

                    if (gz <= gvz) {
                        beta(t + 1) = b;
                        z0 = z1;
                        z1 = z;
                        break;
                    }
                }
                if (beta(t + 1) == 0) {
                    beta(t + 1) = b;
                    z0 = z1;
                    z1 = z;
                }
                delta(t + 2) = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * delta(t + 1) * delta(t + 1)));

                if ((z1 - z0).cwiseAbs().sum() <= 1e-4) {
                    break;
                }
            }
            val.row(i) = z1;
        }
    }
}
#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void bind(py::module_ &m) {
    m.def("LAE", &LAE, "X"_a.noconvert(), "Anchor"_a.noconvert(), "pos"_a.noconvert(), "val"_a.noconvert(),
          "cn"_a = 10);
}
#endif
}  // namespace LAE
