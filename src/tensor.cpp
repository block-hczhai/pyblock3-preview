
#include "tensor.hpp"

py::array_t<double> tensor_transpose(const py::array_t<double> &x,
                                     const py::array_t<int> &perm,
                                     const double alpha, const double beta) {
    int ndim = (int)x.ndim();
    vector<int> shape(ndim);
    for (int i = 0; i < ndim; i++)
        shape[i] = (int)x.shape()[i];
    vector<ssize_t> shape_y(ndim);
    for (int i = 0; i < ndim; i++)
        shape_y[i] = x.shape()[perm.data()[i]];
    py::array_t<double> c(shape_y);
    tensor_transpose_impl(ndim, x.size(), perm.data(), shape.data(), x.data(),
                          c.mutable_data(), alpha, beta);
    return c;
}

py::array_t<double> tensor_tensordot(const py::array_t<double> &a,
                                     const py::array_t<double> &b,
                                     const py::array_t<int> &idxa,
                                     const py::array_t<int> &idxb, double alpha,
                                     double beta) {
    int ndima = (int)a.ndim(), ndimb = (int)b.ndim(), nctr = (int)idxa.size();
    vector<ssize_t> shapec;
    shapec.reserve(ndima + ndimb);
    shapec.insert(shapec.end(), a.shape(), a.shape() + ndima);
    shapec.insert(shapec.end(), b.shape(), b.shape() + ndima);
    for (int i = 0; i < nctr; i++)
        shapec[idxa.data()[i]] = -1, shapec[idxb.data()[i] + ndima] = -1;
    shapec.resize(
        distance(shapec.begin(), remove(shapec.begin(), shapec.end(), -1)));
    py::array_t<double> c(shapec);
    tensordot_impl(a.data(), ndima, a.shape(), b.data(), ndimb, b.shape(), nctr,
              idxa.data(), idxb.data(), c.mutable_data(), alpha, beta);
    return c;
}
