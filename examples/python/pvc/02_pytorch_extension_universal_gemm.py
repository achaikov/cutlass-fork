import torch
from torch.utils.cpp_extension import load
import os

os.environ["CXX"] = "clang++"
CUTLASS_INCLUDE = "" # Provide path to cutlass/include
CUTLASS_UTIL_INCLUDE = "" # Provide path to cutlass/tools/util/include
DPCPP_HOME = os.environ["DPCPP_HOME"]

dtype = torch.float32
print(f"XPU is available {torch.xpu.is_available()}")

cutlass_gemm = load(
    name="cutlass_gemm",
    sources=["pvc_gemm.cpp", "pvc_gemm_kernel.cpp"],
    extra_cflags = [
        "-DCUTLASS_ENABLE_SYCL",
        "-DSYCL_INTEL_TARGET",
        "-DCUTLASS_USE_PACKED_TUPLE=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-std=c++17",
        "-fsycl",
        "-fsycl-targets=intel_gpu_pvc"
    ],
    extra_include_paths=[
        CUTLASS_INCLUDE,
        CUTLASS_UTIL_INCLUDE,
        DPCPP_HOME
        ],
    extra_ldflags=[
        "-fsycl",
        "-fsycl-targets=intel_gpu_pvc",
        f"-L/{DPCPP_HOME}",
    ],
    verbose=True,
)



# Define GEMM inputs
M, N, K, L = 5120, 4096, 4096, 10
alpha, beta = 1.0, 0.0
iterations = 10

# Define GEMM inputs
def initialize(M, N, K, L):
    sizes = [(L, M, K), (L, K, N), (L, M, N), (L, M, N )]
    tensors = [torch.randint(-3, 3, size, device='xpu').to(torch.bfloat16) for size in sizes[:2]]  # A and B
    tensors.append(torch.zeros(sizes[2], dtype=dtype, device='xpu'))  # C
    tensors.append(torch.zeros(sizes[3], dtype=dtype, device='xpu'))  # D
    return tensors

# Utility function to generate `problems` GEMMs of random sizes
def generate_problems(M, N, K, L):
    A, B, C, D = initialize(M, N, K, L)
    return A, B, C, D

A, B, C, D = generate_problems(M, N, K, L)
Ds = cutlass_gemm.run(A, B, C, alpha, beta, iterations)
Ds_torch = A @ B
Ds_torch = Ds_torch.to(dtype)
assert torch.allclose(Ds, Ds_torch) # Fails due to precision
torch.xpu.synchronize()


num_warmup = 20
num_profile = 100

# Warmup iterations
for _ in range(num_warmup):
    Ds = cutlass_gemm.run(A, B, C, alpha, beta, iterations)
    Ds_torch = (A @ B).to(dtype)
    torch.xpu.synchronize()

# Timing iterations
import time
grouped = 0
nongrouped = 0
for _ in range(num_profile):
    start = time.time()
    Ds = cutlass_gemm.run(A, B, C, alpha, beta, iterations)
    torch.xpu.synchronize()
    grouped += time.time() - start

    start = time.time()
    Ds_torch = (A @ B).to(dtype)
    torch.xpu.synchronize()
    nongrouped += time.time() - start

print('Grouped:     {:.3f} us'.format(grouped * 1e6/num_profile))
print('Non-Grouped: {:.3f} us'.format(nongrouped * 1e6/num_profile))
print('Speedup: {:.3f}'.format(nongrouped / grouped))