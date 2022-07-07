//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define BLOCK_DIAG_MAT_VEC_DATA_SETUP_HIP                               \
  const Index_type N = m_N;                                             \
  allocAndInitHipDeviceData(Me, m_Me, ndof*ndof*N);                    \
  allocAndInitHipDeviceData(X, m_X, ndof*N);                           \
  allocAndInitHipDeviceData(Y, m_Y, ndof*N);

#define BLOCK_DIAG_MAT_VEC_DATA_TEARDOWN_HIP                            \
  getHipDeviceData(m_Me, Me, ndof*ndof*N);                             \
  getHipDeviceData(m_X, X, ndof*N);                                          \
  getHipDeviceData(m_Y, Y, ndof*N);                                          \
  deallocHipDeviceData(Me);                                             \
  deallocHipDeviceData(X);                                              \
  deallocHipDeviceData(Y);

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void block_diag_mat_vec(const Index_type NE, const Index_type ndof,
								   const Real_ptr X, const Real_ptr Me, Real_ptr Y){
	const Index_type tid = threadIdx.x + blockIdx.x * blockDim.x;
	const Index_type num_blocks = (ndof + block_size - 1)/ block_size;

	__shared__ Real_type Xs[block_size];
	Real_type dot = 0.0;

	for ( Index_type c = 0; c < num_blocks; ++c)
	{
		if ((c * block_size + threadIdx.x) <  ndof) 
			Xs[threadIdx.x] = X[threadIdx.x + c * block_size];
		else
			Xs[threadIdx.x] = 0.0;
		__syncthreads();

		for (Index_type r = 0; r < block_size; ++r) {
			dot += Me[tid + (r + block_size * c) * ndof] * Xs[r];
		}
		__syncthreads();
	}
	if (tid < ndof)
		Y[tid] = dot;
}	

template < size_t block_size >
void BLOCK_DIAG_MAT_VEC::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  //const Index_type N = getActualProblemSize();
  const Index_type N = 1;
  constexpr Index_type ndof = m_ndof;
//  printf("N = %d\n", N);
  dim3 blockDim(block_size);
  dim3 gridDim(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, block_size)));

  BLOCK_DIAG_MAT_VEC_DATA_SETUP;

  BLOCK_DIAG_MAT_VEC_DATA_INIT;

  if (vid == Base_HIP) {

    BLOCK_DIAG_MAT_VEC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
     hipLaunchKernelGGL((block_diag_mat_vec<block_size>), dim3(gridDim), dim3(blockDim), 0, 0, N, ndof, X, Me, Y);
     hipErrchk( hipGetLastError() );	
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_DATA_TEARDOWN_HIP;
    for(int i = 0; i != ndof; ++i){
      for(int j = 0; j != N; ++j){ 
          printf("Y[%d][%d] = %f\n", i, j, m_Y[i + ndof * j]);
      }
    }
  } else if (vid == Lambda_HIP) {

    BLOCK_DIAG_MAT_VEC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    BLOCK_DIAG_MAT_VEC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    BLOCK_DIAG_MAT_VEC_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  BLOCK_DIAG_MAT_VEC : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(BLOCK_DIAG_MAT_VEC, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
