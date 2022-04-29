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
  constexpr Index_type Ne = m_Ne;                 \
  constexpr Index_type NeNe = m_Ne * m_Ne;                 \
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

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void mat_fused_mul_add(const Real_ptr A, const Real_ptr B, Real_ptr D,
                                  Index_type N){
constexpr Index_type Ne = 16;                                  
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
  // compute a 16x16x16 matrix multiplication using a single wavefront.
#if defined(RP_USE_DOUBLE)
  using double4 = __attribute__((__vector_size__(4 * sizeof(double)))) double;
  double4 result = {0};
#elif defined(RP_USE_FLOAT)
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 result = {0};
#endif
  Index_type a_idx = Ne * threadIdx.x + threadIdx.y + ii*(Ne*Ne);
  Index_type b_idx = threadIdx.x + Ne * threadIdx.y + ii*(Ne*Ne);

  for(int i = 0; i < 4; ++i){
    Real_type a = A[a_idx];
    Real_type b = B[b_idx];

#ifdef __gfx90a__	
#if defined(RP_USE_DOUBLE)
	result = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, result, 0, 0, 0);
#elif defined(RP_USE_FLOAT)
    result = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, result, 0, 0, 0);
#endif  
#endif
    a_idx += 4; // move four columns to the right
    b_idx += 4*Ne; // move four rows down
  }

  #pragma unroll 4
  for(Index_type i = 0; i < 4; ++i){
    const Index_type d_idx =  threadIdx.x            
                     + i * Ne                
                     + threadIdx.y * 4 * Ne
                     + ii*(Ne*Ne); 

    D[d_idx] = result[i];
  }
}
}


template < size_t block_size >
void MAT_FUSED_MUL_ADD::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  constexpr Index_type Ne = m_Ne;
  constexpr Index_type NeNe = m_Ne * m_Ne;

  dim3 gridDim (1, 1, 1);
  dim3 blockDim(Ne, 4, 1);

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  if (vid == Base_HIP) {

	for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
  		for(Index_type i = 0; i != NeNe; ++i){ m_A[i+(ii*NeNe)] = i; }
  		for(Index_type i = 0; i != NeNe; ++i){ m_B[i+(ii*NeNe)] = NeNe - 1 - i; }
	}
    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((mat_fused_mul_add<block_size>), dim3(gridDim), dim3(blockDim), 0, 0,
                         A, B, D, N);
      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_FUSED_MUL_ADD, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
