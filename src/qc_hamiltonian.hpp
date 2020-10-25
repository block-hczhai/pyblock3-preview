
#ifdef I
#undef I
#endif
#include <pybind11/numpy.h>
#include <tuple>

namespace py = pybind11;
using namespace std;

vector<tuple<py::array_t<uint32_t>, py::array_t<uint32_t>, py::array_t<double>,
             py::array_t<uint32_t>>>
build_qc_mpo(py::array_t<int32_t> orb_sym, py::array_t<double> t,
             py::array_t<double> v);
