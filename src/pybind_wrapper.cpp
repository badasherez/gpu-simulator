#include <torch/extension.h>
#include "../include/Hopper_simulator.h"
#include "../include/Ampere_simulator.h"
#include "../utils/utils.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU Simulator Python Bindings (Hopper & Ampere)"; // Optional module docstring
    
    // Bind the Hopper_simulator class
    py::class_<Hopper_simulator>(m, "Hopper_simulator")
        .def(py::init<>())
        .def("matmul_bfloat16", &Hopper_simulator::matmul_bfloat16,
             "Perform matrix multiplication using bfloat16 to Gfloat conversion",
             py::arg("A"), py::arg("B"))
        .def("matmul_float16", &Hopper_simulator::matmul_float16,
             "Perform matrix multiplication using float16 to Gfloat conversion",
             py::arg("A"), py::arg("B"))
        .def("matmul", &Hopper_simulator::matmul,
             "Perform matrix multiplication with automatic dtype dispatch (bfloat16 or float16)",
             py::arg("A"), py::arg("B"))
        .def("group_sum", &Hopper_simulator::group_sum,
             "Compute group sum of Gfloat array",
             py::arg("array"));

    // Bind the Ampere_simulator class
    py::class_<Ampere_simulator>(m, "Ampere_simulator")
        .def(py::init<>())
        .def("matmul_bfloat16", &Ampere_simulator::matmul_bfloat16,
             "Perform matrix multiplication using bfloat16 to Gfloat conversion",
             py::arg("A"), py::arg("B"))
        .def("matmul_float16", &Ampere_simulator::matmul_float16,
             "Perform matrix multiplication using float16 to Gfloat conversion",
             py::arg("A"), py::arg("B"))
        .def("matmul", &Ampere_simulator::matmul,
             "Perform matrix multiplication with automatic dtype dispatch (bfloat16 or float16)",
             py::arg("A"), py::arg("B"))
        .def("group_sum", &Ampere_simulator::group_sum,
             "Compute group sum of Gfloat array",
             py::arg("array"));
    

    
    // Bind utility functions
    m.def("bfloat16_to_gfloat_tensor", &bfloat16_to_gfloat_tensor,
          "Convert bfloat16 tensor to Gfloat tensor",
          py::arg("tensor"));
} 