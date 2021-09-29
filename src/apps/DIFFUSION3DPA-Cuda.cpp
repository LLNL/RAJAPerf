//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define DIFFUSION3DPA_DATA_SETUP_CUDA                                        \
  //  allocAndInitCudaDeviceData(B, m_B, DPA_Q1D *DPA_D1D);                     \
  //  allocAndInitCudaDeviceData(Bt, m_Bt, DPA_Q1D *DPA_D1D);                   \
  allocAndInitCudaDeviceData(D, m_D, DPA_Q1D *DPA_Q1D *DPA_Q1D *m_NE);              \
  allocAndInitCudaDeviceData(X, m_X, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);              \
  allocAndInitCudaDeviceData(Y, m_Y, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);

#define DIFFUSION3DPA_DATA_TEARDOWN_CUDA                                      \
  //  getCudaDeviceData(m_Y, Y, DPA_D1D *DPA_D1D *DPA_D1D *m_NE);                   \
  //  deallocCudaDeviceData(B);                                         \
  //  deallocCudaDeviceData(Bt);                                        \
  //  deallocCudaDeviceData(D);                                         \
  //  deallocCudaDeviceData(X);                                         \
  //deallocCudaDeviceData(Y);

//#define USE_RAJA_UNROLL
#define RAJA_DIRECT_PRAGMA(X) _Pragma(#X)
#if defined(USE_RAJA_UNROLL)
#define RAJA_UNROLL(N) RAJA_DIRECT_PRAGMA(unroll(N))
#else
#define RAJA_UNROLL(N)
#endif
#define FOREACH_THREAD(i, k, N)                                                \
  for (int i = threadIdx.k; i < N; i += blockDim.k)

__global__ void Diffusion3DPA(Index_type NE, const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D, const Real_ptr X, Real_ptr Y) {

  const int e = blockIdx.x;

}

void DIFFUSION3DPA::runCudaVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    DIFFUSION3DPA_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(DPA_Q1D, DPA_Q1D, 1);

      //      Diffusion3DPA<<<NE, nthreads_per_block>>>(NE, B, Bt, D, X, Y);

      cudaErrchk( cudaGetLastError() );
    }
    stopTimer();

    DIFFUSION3DPA_DATA_TEARDOWN_CUDA;

    break;
  }

  case RAJA_CUDA: {

    DIFFUSION3DPA_DATA_SETUP_CUDA;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
                                                   ,RAJA::expt::cuda_launch_t<true>
                                                   >;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::cuda_block_x_direct
                                           >;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::cuda_thread_x_loop
                                             >;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::cuda_thread_y_loop
                                             >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {


    }  // loop over kernel reps
    stopTimer();

    DIFFUSION3DPA_DATA_TEARDOWN_CUDA;

    break;
  }

  default: {

    std::cout << "\n DIFFUSION3DPA : Unknown Cuda variant id = " << vid << std::endl;
    break;
  }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
