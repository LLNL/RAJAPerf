//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

#define MAT_FUSED_MUL_ADD_DATA_SETUP_HIP           \
  const Index_type N = m_N;                        \
  allocAndInitHipDeviceData(A, m_A, N);            \
  allocAndInitHipDeviceData(B, m_B, N);            \
  allocAndInitHipDeviceData(D, m_D, N);			   

#define MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP        \
  getHipDeviceData(m_A, A, N);                     \
  getHipDeviceData(m_B, B, N);                     \
  getHipDeviceData(m_D, D, N);                     \
  deallocHipDeviceData(A);                         \
  deallocHipDeviceData(B);                         \
  deallocHipDeviceData(D);						   

__global__ void mat_fused_mul_add_builtin(const Real_ptr A, const Real_ptr B, Real_ptr D, const Index_type N){ 
  constexpr Index_type Ne = 16;
  for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
  using real4 = __attribute__((__vector_size__(4 * sizeof(Real_type)))) Real_type;
  real4 result = {0};

  Index_type a_idx = Ne * threadIdx.x + threadIdx.y + ii*(Ne*Ne);
  Index_type b_idx = threadIdx.x + Ne * threadIdx.y + ii*(Ne*Ne);

  for(Index_type i = 0; i < 4; ++i){
    Real_type a = A[a_idx];
    Real_type b = B[b_idx];

#ifdef __gfx90a__	
#if defined(RP_USE_DOUBLE)
    result = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, result, 0, 0, 0);
#elif defined(RP_USE_FLOAT)
    result = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, result, 0, 0, 0);
#endif  
#endif

#ifdef __gfx908__
#if defined(RP_USE_FLOAT)
    result = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, result, 0, 0, 0);
#endif  
#endif
    a_idx += 4; // move two columns to the right
    b_idx += 4*Ne; // move two rows down
  }

  #pragma unroll 4
  for(Index_type i = 0; i < 4; ++i){
    const Index_type d_idx =  threadIdx.x
      + Ne * (threadIdx.y + 4 * i); 
    D[d_idx + ii*(Ne*Ne)] = result[i];
  }
}
}
//Reference for cases with no hardware support
template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void mat_fused_mul_add(const Real_ptr A, const Real_ptr B, Real_ptr D,
                                  const Index_type N){
  constexpr Index_type Ne = 16;
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){  
  Index_type col = threadIdx.x + blockIdx.x * blockDim.x; 
  Index_type row = threadIdx.y + blockIdx.y * blockDim.y; 
  MAT_FUSED_MUL_ADD_BODY;
}
}
template <  Index_type block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void mat_fused_lam(const Index_type N, Lambda body)
{
  constexpr Index_type Ne = 16;
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){  
  Index_type col = threadIdx.x + blockIdx.x * blockDim.x; 
  Index_type row = threadIdx.y + blockIdx.y * blockDim.y; 
     body(ii,col,row);
   }
}
void MAT_FUSED_MUL_ADD::runHipVariantBuiltin(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();  
  const Index_type N = m_N;
  constexpr Index_type Ne = m_Ne;
  constexpr Index_type NeNe = m_Ne * m_Ne;

  dim3 gridDim (1, 1, 1);
  dim3 blockDimBuiltin(Ne, 4, 1);

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;

  if (vid == Base_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      hipLaunchKernelGGL((mat_fused_mul_add_builtin), dim3(gridDim), dim3(blockDimBuiltin), 0, 0, A, B, D, iend);
      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;
  
  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Hip variant id = " << vid
              << std::endl;
  }
}
template < size_t block_size >
void MAT_FUSED_MUL_ADD::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();  
  const Index_type N = m_N;
  constexpr Index_type Ne = m_Ne;
  constexpr Index_type NeNe = m_Ne * m_Ne;

  constexpr Index_type tile_size = gpu_block_size::sqrt(block_size);
  dim3 blockDim(tile_size, tile_size);
  dim3 gridDim(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(Ne, block_size)),
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(Ne, block_size)), 
               static_cast<size_t>(1));
  MAT_FUSED_MUL_ADD_DATA_SETUP;

  MAT_FUSED_MUL_ADD_DATA_INIT;

  if (vid == Base_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      hipLaunchKernelGGL((mat_fused_mul_add<block_size>), dim3(gridDim), dim3(blockDim), 0, 0, A, B, D, iend);
      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;
  } else if (vid == Lambda_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto mat_fused_lamda =
        [=] __device__ (Index_type ii, Index_type row, Index_type col) {
            MAT_FUSED_MUL_ADD_BODY;
        };        
      hipLaunchKernelGGL((mat_fused_lam<block_size, decltype(mat_fused_lamda)>),
                         dim3(gridDim), dim3(blockDim), 0, 0,
                         iend, mat_fused_lamda);      
    hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    RAJA::RangeSegment ii_range(0, (N/(Ne*Ne)));
    RAJA::RangeSegment row_range(0, Ne);
    RAJA::RangeSegment col_range(0, Ne);

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernel<
        RAJA::statement::For<2, RAJA::loop_exec,
          RAJA::statement::Tile<1, RAJA::tile_fixed<block_size>, RAJA::hip_block_y_loop,
            RAJA::statement::Tile<0, RAJA::tile_fixed<block_size>, RAJA::hip_block_x_loop,
              RAJA::statement::For<1, RAJA::hip_thread_y_direct,
                RAJA::statement::For<0, RAJA::hip_thread_x_direct,
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

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Hip variant id = " << vid
              << std::endl;
  }
}
std::string getArch()
{  
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::string gcnArchName(devProp.gcnArchName);
  std::string hipArch = gcnArchName.substr(0, 6);
  return hipArch;
}
bool builtinSupported()
{
	std::string hipArch = getArch();
#if defined(RP_USE_DOUBLE)
	if (hipArch=="gfx90a") 
		return true;
#endif
#if defined(RP_USE_FLOAT)
	if (hipArch=="gfx90a" || hipArch=="gfx908")
		return true;
#endif
return false;
}
void MAT_FUSED_MUL_ADD::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool builtin_supported = builtinSupported();

   size_t t = 0;
  if ( vid == Base_HIP  && builtin_supported) {

    if (tune_idx == t) {

      runHipVariantBuiltin(vid);

    }

    t += 1;
  }
  if ( (vid == Base_HIP) || (vid == RAJA_HIP) || (vid == Lambda_HIP)){

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0 ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantImpl<block_size>(vid);

        }

        t += 1;

      }

    });
  }
  else {

    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Hip variant id = " << vid << std::endl;

  }

}

void MAT_FUSED_MUL_ADD::setHipTuningDefinitions(VariantID vid)
{
  bool builtin_supported = builtinSupported();
  if ( vid == Base_HIP ) {

  if (builtin_supported) {
    addVariantTuningName(vid, "builtin");
  }
  }
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "block_"+std::to_string(block_size));
      }

    });

}

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
