#include <torch/extension.h>
#include <ATen/ATen.h>


#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "common.h"
#include <sycl/sycl.hpp>

#include <CL/sycl.hpp>
#include <vector>


using namespace cute;

template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;


  ProblemShapeType problem_size;

  
  //
  // Methods
  //

  void check_tensor_device(const at::Tensor& tensor) {
      // Check if the tensor is defined
      if (!tensor.defined()) {
          std::cout << "Tensor is undefined." << std::endl;
          return;
      }

      // Get the device of the tensor
      auto device = tensor.device();

      // Print the device
      if (device.is_cpu()) {
          std::cout << "Tensor is on CPU." << std::endl;
      } else if (device.is_cuda()) {
          std::cout << "Tensor is on CUDA device." << device.index() << std::endl;
      } else if (device.is_xpu()) {
          std::cout << "Tensor is on XPU device." << device.index() << std::endl;
      } else {
          std::cout << "Tensor is on an unknown device." << std::endl;
      }
  }

  void check_tensor_stride(const at::Tensor& tensor) {
    if (!tensor.defined()) {
        std::cout << "Tensor is undefined." << std::endl;
        return;
    }

    std::cout << "Tensor strides: ";
    for (auto stride : tensor.strides()) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
}


 at::Tensor run(const cutlass::KernelHardwareInfo& hw_info,
                  const at::Tensor& A,
                  const at::Tensor& B,
                  const at::Tensor& C,
                  float alpha, float beta, int iterations) {

    int L = A.sizes()[0];
    int M = A.sizes()[1];
    int K = A.sizes()[2];
    int N = B.sizes()[2];
    
    
    problem_size = ProblemShapeType{M, N, K, L};

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(L, M, K));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(L, K, N));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(L, M, N));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(L, M, N));
    
    at::Tensor D = C.new_empty({L, M, N}, torch::kF32);

    cutlass::bfloat16_t* A_ptr = reinterpret_cast<cutlass::bfloat16_t*>(A.contiguous().data_ptr<torch::BFloat16>());
    cutlass::bfloat16_t* B_ptr = reinterpret_cast<cutlass::bfloat16_t*>(B.contiguous().data_ptr<torch::BFloat16>());
    float* C_ptr = C.contiguous().data_ptr<float>();
    float* D_ptr = D.contiguous().data_ptr<float>();
    
    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {A_ptr, stride_A, B_ptr, stride_B},
      {{alpha, beta}, C_ptr, stride_C, D_ptr, stride_D},
      hw_info
    };
    Gemm gemm_op;


    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm_op.can_implement(arguments);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    gemm_op.run();

    syclcompat::wait();

    return D;
  }
};

at::Tensor run_cutlass_gemm(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& C,
  float alpha, float beta,
  int iterations) {

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    bool passed;

    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = float;                   // <- data type of accumulator
    using ElementComputeEpilogue = float;  // <- data type of epilogue operations
    using ElementInputA = bfloat16_t;                        // <- data type of elements in input matrix A
    using ElementInputB = bfloat16_t;                        // <- data type of elements in input matrix B
    using ElementOutput = float;                        // <- data type of elements in output matrix D

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
    using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

    // Workgroup-level tile
    using TileShape = Shape<_256, _256, _32>;
    using TiledMma = TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
            Layout<Shape<_8,_4,_1>>,
            Tile<_64,_64,_32>>; // Subgroup level-tile

    constexpr int PipelineStages = 3;
    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

    using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
            ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

    using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
            decltype(tile_shape(TiledMma()))>;
    using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
            EpilogueDispatchPolicy,
            TileShape,
            ElementAccumulator,
            cutlass::gemm::TagToStrideC_t<LayoutC>,
            ElementOutput,
            cutlass::gemm::TagToStrideC_t<LayoutD>,
            FusionCallBacks,
            XE_2D_U32x8x16_LD_N,
            void, void,
            XE_2D_U32x8x16_ST_N,
            void, void>;

    // Mainloop
    using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            GEMMDispatchPolicy,
            TileShape,
            ElementInputA,
            cutlass::gemm::TagToStrideA_t<LayoutA>,
            ElementInputB,
            cutlass::gemm::TagToStrideB_t<LayoutB>,
            TiledMma,
            GmemTiledCopyA, void, void, cute::identity,  // A
            GmemTiledCopyB, void, void, cute::identity   // B
    >;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    ExampleRunner<Gemm> runner;

    torch::Tensor result_tensor = runner.run(hw_info, A, B, C, alpha, beta, iterations);
    return result_tensor;
}
