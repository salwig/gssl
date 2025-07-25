/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#include <Eigen/IterativeLinearSolvers>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Variational.h"

class VGL {
   public:
    size_t C;
    size_t C_active;
    size_t K;
    Vector<bool> Mask;
    Variational &em;

    precision_t eps_;  //= std::numeric_limits<precision_t>::min();
    bool raise_warning;

    std::vector<std::vector<q_t>> all_list;
    std::vector<std::vector<size_t>> W_ind;
    std::vector<std::vector<precision_t>> W_val;
    Vector<size_t> idx;

    SparseMatrix<> W;
    SparseMatrix<> W_l;
    SparseMatrix<> L;
    SparseMatrix<> I;
    SparseMatrix<> B;
    ColMatrix<> soft_class_matrix;

    VGL(Variational &, size_t, size_t, size_t, Vector<bool>, precision_t);

    void get_partition();
    void get_partition_indices(cRef<Vector<size_t>>);
    void setFromTriplets(SparseMatrix<> &, size_t);
    void setFromTriplets_B(SparseMatrix<> &, size_t);
    size_t SpMM(bool);
    size_t compute_B(cRef<Vector<long>>);

    void compute_L();
    void compute_W(bool);
    void compute_soft_class_matrix(cRef<Vector<long>>, size_t, precision_t, const size_t &,
                                   const precision_t &);
    void ConjugateGradient(const size_t &, const precision_t &);
    void predict(Ref<Matrix<>>, Ref<Vector<size_t>>, const bool &);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
    static void bind(py::module_ &m);
#endif
};

VGL::VGL(Variational &_em, size_t _C, size_t _C_active, size_t _K, Vector<bool> _mask, precision_t _eps) :
    C(_C),
    C_active(_C_active),
    K(_K),
    Mask(_mask),
    em(_em),
    raise_warning(true),
    eps_(_eps),
    all_list(get_max_threads()),
    W_ind(_C),
    W_val(_C),
    idx(_C),
    W(_C_active, _C_active),
    W_l(_C_active, _C_active),
    L(_C_active, _C_active),
    I(_C_active, _C_active),
    B(_C_active, _K),
    soft_class_matrix(_C_active, _K) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        all_list[thread_num] = std::vector<q_t>(_C);
    }
    I.setIdentity();
    size_t i = 0;
    // Vector<size_t> rev_idx(C_active);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            idx[c] = i;
            // rev_idx[i] = c;
            i++;
        } else {
            idx[c] = -1;
        }
    }
}

void VGL::get_partition() {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        std::fill(all_list[thread_num].begin(), all_list[thread_num].end(), q_t(0));
#pragma omp for
        for (size_t n = 0; n < em.qs.size(); n++) {
            for (const auto &[c, val] : em.qs[n]) {
                if (val > 0. and Mask[c]) {  // ignore if q rounds to zero
                    all_list[thread_num][c].emplace_back(n, val);
                }
            }
        }
    }
}

void VGL::get_partition_indices(cRef<Vector<size_t>> Indices) {
#pragma omp parallel
    {
        size_t thread_num = get_thread_num();
        std::fill(all_list[thread_num].begin(), all_list[thread_num].end(), q_t(0));
#pragma omp for
        for (size_t n : Indices) {
            for (const auto &[c, val] : em.qs[n]) {
                if (val > 0. and Mask[c]) {  // ignore if q rounds to zero
                    all_list[thread_num][c].emplace_back(n, val);
                }
            }
        }
    }
}

void VGL::setFromTriplets(SparseMatrix<> &Matrix, size_t nonZeros) {
    std::vector<Eigen::Triplet<precision_t>> coeff;
    coeff.reserve(nonZeros);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            for (size_t j = 0; j < W_ind[c].size(); j++) {
                if (W_val[c][j] > eps_) {
                    coeff.emplace_back(idx[c], idx[W_ind[c][j]], W_val[c][j]);
                } else {
                    coeff.emplace_back(idx[c], idx[W_ind[c][j]], eps_);
                }
            }
        }
    }
    Matrix.setFromTriplets(coeff.begin(), coeff.end());
}

void VGL::setFromTriplets_B(SparseMatrix<> &Matrix, size_t nonZeros) {
    std::vector<Eigen::Triplet<precision_t>> coeff;
    coeff.reserve(nonZeros);
    for (size_t c = 0; c < C; c++) {
        if (Mask[c]) {
            for (size_t j = 0; j < W_ind[c].size(); j++) {
                coeff.emplace_back(idx[c], W_ind[c][j], W_val[c][j]);
            }
        }
    }
    Matrix.setFromTriplets(coeff.begin(), coeff.end());
}

size_t VGL::SpMM(bool allow_self_loops) {
    size_t nonZeros = 0;
#pragma omp parallel
    {
        map_t<size_t, size_t> index_map;

#pragma omp for schedule(dynamic)
        for (size_t c = 0; c < C; c++) {
            W_ind[c].clear();
            W_ind[c].reserve(C);

            W_val[c].clear();
            W_val[c].reserve(C);

            index_map.clear();

            if (Mask[c]) {
                for (const auto &list : all_list) {
                    if (!list[c].empty()) {
                        for (const auto &[n, log_prob_c] : list[c]) {
                            for (const auto &[k, log_prob] : em.qs[n]) {
                                if ((Mask[k])) {
                                    if ((c != k) or (allow_self_loops)) {
                                        auto el = index_map.find(k);
                                        if (el != index_map.end()) {
                                            W_val[c][el->second] += log_prob_c * log_prob;
                                        } else {
                                            index_map.emplace(k, W_ind[c].size());
                                            W_ind[c].push_back(k);
                                            W_val[c].push_back(log_prob_c * log_prob);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
#pragma omp atomic
            nonZeros += index_map.size();
        }
    }
    return nonZeros;
}

size_t VGL::compute_B(cRef<Vector<long>> labels) {
    size_t nonZeros = 0;

#pragma omp parallel
    {
        map_t<size_t, size_t> index_map;
        long k;

#pragma omp for schedule(dynamic)
        for (size_t c = 0; c < C; c++) {
            W_ind[c].clear();
            W_ind[c].reserve(C);

            W_val[c].clear();
            W_val[c].reserve(C);

            index_map.clear();

            if (Mask[c]) {
                for (const auto &list : all_list) {
                    if (!list[c].empty()) {
                        for (const auto &[n, log_prob_c] : list[c]) {
                            k = labels[n];
                            auto el = index_map.find(k);
                            if (el != index_map.end()) {
                                W_val[c][el->second] += log_prob_c;
                            } else {
                                index_map.emplace(k, W_ind[c].size());
                                W_ind[c].push_back(k);
                                W_val[c].push_back(log_prob_c);
                            }
                        }
                    }
                }
            }
#pragma omp atomic
            nonZeros += index_map.size();
        }
    }
    return nonZeros;
}

void VGL::compute_W(bool allow_self_loops) {
    get_partition();
    size_t nonZeros = SpMM(allow_self_loops);
    setFromTriplets(W, nonZeros);
    return;
}

void VGL::compute_L() {
    Vector<> D(C_active);
    D.fill(0);
#pragma omp parallel for
    for (size_t c = 0; c < C_active; c++) {
        for (SparseMatrix<>::InnerIterator it(W, c); it; ++it) {
            D[c] += it.value();
        }
        if (D[c] == 0.0) {
            D[c] = 1.0;
        }
        D[c] = 1. / sqrt(D[c]);
    }

    L = I;
    L -= D.asDiagonal() * W * D.asDiagonal();
    L = L.pruned();
    return;
}

void VGL::compute_soft_class_matrix(cRef<Vector<long>> labels, size_t label_size, precision_t gamma,
                                    const size_t &cg_maxiter, const precision_t &cg_rtol) {
    Vector<size_t> idx_l(label_size);
    size_t i = 0;
    size_t N = labels.size();
    for (size_t n = 0; n < N; n++) {
        if (labels[n] != (-1)) {
            idx_l[i] = n;
            i++;
        }
    }
    get_partition_indices(idx_l);
    size_t nonZeros = SpMM(true);
    setFromTriplets(W_l, nonZeros);
    W_l += gamma * L + 1e-6 * I;

    nonZeros = compute_B(labels);
    setFromTriplets_B(B, nonZeros);

    ConjugateGradient(cg_maxiter, cg_rtol);
    return;
}

void VGL::ConjugateGradient(const size_t &cg_maxiter, const precision_t &cg_rtol) {
    Eigen::ConjugateGradient<SparseMatrix<>, Eigen::Lower | Eigen::Upper> solver;

    solver.setMaxIterations(cg_maxiter);
    solver.setTolerance(cg_rtol);

    solver.compute(W_l);
    if (solver.info() != Eigen::Success) {
        std::cerr << "WARNING: ConjugateGradient: decomposition failed, "
                     "try to increase 'cg_maxiter' or lower 'cg_rtol'\n";
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t k = 0; k < K; k++) {
        soft_class_matrix.col(k) = solver.solve(B.col(k));
    }
    return;
}

void VGL::predict(Ref<Matrix<>> prop_labels, Ref<Vector<size_t>> predicted_labels, const bool &class_norm) {
    prop_labels.fill(0);
    if (class_norm) {
        Vector<precision_t> normalizer(K);
        normalizer.fill(0);
#pragma omp parallel
        {
            size_t ind;
            Vector<precision_t> thread_normalizer(K);
            thread_normalizer.fill(0);

#pragma omp for
            for (size_t n = 0; n < em.qs.size(); n++) {
                for (size_t k = 0; k < K; k++) {
                    for (const auto &[c, log_prob] : em.qs[n]) {
                        if (Mask[c]) {
                            prop_labels(n, k) += log_prob * soft_class_matrix(idx[c], k);
                        }
                    }
                }
                thread_normalizer += prop_labels.row(n);
            }
#pragma omp critical
            normalizer += thread_normalizer;

#pragma omp barrier
#pragma omp for
            for (size_t n = 0; n < em.qs.size(); n++) {
                prop_labels.row(n).array() /= normalizer.array();
                prop_labels.row(n).maxCoeff(&ind);
                predicted_labels[n] = ind;
            }
        }
    } else {
#pragma omp parallel
        {
            size_t ind;

#pragma omp for
            for (size_t n = 0; n < em.qs.size(); n++) {
                for (size_t k = 0; k < K; k++) {
                    for (const auto &[c, log_prob] : em.qs[n]) {
                        if (Mask[c]) {
                            prop_labels(n, k) += log_prob * soft_class_matrix(idx[c], k);
                        }
                    }
                }
                prop_labels.row(n).maxCoeff(&ind);
                predicted_labels[n] = ind;
            }
        }
    }

    return;
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE

void VGL::bind(py::module_ &m) {
    py::class_<VGL> VGL_class_(m, "VGL", py::module_local());

    VGL_class_.def(py::init<Variational &, size_t, size_t, size_t, Vector<bool>, precision_t>(), "em"_a, "C"_a,
                   "C_active"_a, "K"_a, "mask"_a, "eps"_a);

    VGL_class_.def_readonly("W", &VGL::W);
    VGL_class_.def_readonly("W_l", &VGL::W_l);
    VGL_class_.def_readonly("L", &VGL::L);
    VGL_class_.def_readonly("B", &VGL::B);
    VGL_class_.def_readonly("soft_class_matrix", &VGL::soft_class_matrix);

    VGL_class_.def("compute_W", &VGL::compute_W, "allow_self_loops"_a = true);
    VGL_class_.def("compute_L", &VGL::compute_L);

    VGL_class_.def("compute_soft_class_matrix", &VGL::compute_soft_class_matrix, "labels"_a.noconvert(),
                   "label_size"_a, "gamma"_a, "cg_maxiter"_a = 20,
                   "cg_rtol"_a = 1e-6);  //
    VGL_class_.def("predict", &VGL::predict, "prop_labels"_a.noconvert(), "predicted_labels"_a.noconvert(),
                   "class_norm"_a = true);
}
#endif
