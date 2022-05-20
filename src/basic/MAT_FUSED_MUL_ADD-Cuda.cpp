//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

#define MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA           \
  const Index_type N = m_N;                         \
  const Index_type Ne = m_Ne;                       \
  allocAndInitCudaDeviceData(A, m_A, N);            \
  allocAndInitCudaDeviceData(B, m_B, N);            \
  allocAndInitCudaDeviceData(D, m_D, N);			

#define MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA        \
  getCudaDeviceData(m_A, A, N);                     \
  getCudaDeviceData(m_B, B, N);                     \
  getCudaDeviceData(m_D, D, N);                     \
  deallocCudaDeviceData(A);                         \
  deallocCudaDeviceData(B);                         \
  deallocCudaDeviceData(D);							


template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void mat_fused_mul_add(const Real_ptr A, const Real_ptr B, Real_ptr D,
                                  Index_type N){
  constexpr Index_type Ne = 16;
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
  Index_type col = threadIdx.x + blockIdx.x * blockDim.x + ii*(Ne*Ne);
  Index_type row = threadIdx.y + blockIdx.y * blockDim.y + ii*(Ne*Ne);

  MAT_FUSED_MUL_ADD_BODY;
}
}
template <  Index_type block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void mat_fused_lam(Index_type N, Lambda body)
{
  constexpr Index_type Ne = 16;
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){  
  Index_type col = threadIdx.x + blockIdx.x * blockDim.x; 
  Index_type row = threadIdx.y + blockIdx.y * blockDim.y; 
     body(ii,col,row);
   }
}
template < size_t block_size >
void MAT_FUSED_MUL_ADD::runCudaVariantImpl(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  constexpr Index_type Ne = m_Ne;
  constexpr Index_type NeNe = m_Ne * m_Ne;

  dim3 gridDim (1, 1, 1);
  dim3 blockDim(Ne, Ne, 1);
  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;

  if (vid == Base_CUDA) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      mat_fused_mul_add<block_size><<<dim3(gridDim), dim3(blockDim)>>>(A, B, D, N);
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;

  } else if (vid == Lambda_CUDA) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto mat_fused_lamda =
        [=] __device__ (Index_type ii, Index_type row, Index_type col) {
            MAT_FUSED_MUL_ADD_BODY;
        };        
      mat_fused_lam<block_size, decltype(mat_fused_lamda)>
                    <<<dim3(gridDim), dim3(blockDim)>>>(N, mat_fused_lamda);
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {
    dim3 gridDim (1, 1, 1);
    dim3 blockDim(Ne, Ne, 1);

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    RAJA::RangeSegment row_range(0, Ne);
    RAJA::RangeSegment col_range(0, Ne);
    RAJA::RangeSegment ii_range(0, (N/(Ne*Ne)));
    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernel<
        RAJA::statement::For<2, RAJA::loop_exec,
          RAJA::statement::Tile<1, RAJA::tile_fixed<block_size>, RAJA::cuda_block_y_loop,
            RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>, RAJA::cuda_block_x_loop,
              RAJA::statement::For<1, RAJA::cuda_thread_y_direct,
                RAJA::statement::For<0, RAJA::cuda_thread_x_direct,
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
        >
      >;
      RAJA::kernel<EXEC_POL>(RAJA::make_tuple(row_range, col_range, ii_range),
    [=] RAJA_DEVICE (Index_type row, Index_type col, Index_type ii) {
        MAT_FUSED_MUL_ADD_BODY;
        });
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;


  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_FUSED_MUL_ADD, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
