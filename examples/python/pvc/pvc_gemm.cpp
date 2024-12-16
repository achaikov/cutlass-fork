#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>



at::Tensor run_cutlass_gemm(const at::Tensor A,
  const at::Tensor B,
  const at::Tensor C,
  float alpha, float beta, int iterations);

// C++ interface
at::Tensor call_cutlass_gemm(const at::Tensor A,
  const at::Tensor B,
  const at::Tensor C,
  float alpha, float beta, int iterations) {
  return run_cutlass_gemm(A, B, C, alpha, beta, iterations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", py::overload_cast<const at::Tensor, const at::Tensor, const at::Tensor, float, float, int>(&call_cutlass_gemm),
        py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("alpha") = 1.f, py::arg("beta") = 0.f, py::arg("iterations") = 1);
}
