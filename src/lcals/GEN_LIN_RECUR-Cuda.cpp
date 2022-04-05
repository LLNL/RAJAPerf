//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define GEN_LIN_RECUR_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(b5, m_b5, m_N); \
  allocAndInitCudaDeviceData(stb5, m_stb5, m_N); \
  allocAndInitCudaDeviceData(sa, m_sa, m_N); \
  allocAndInitCudaDeviceData(sb, m_sb, m_N);

#define GEN_LIN_RECUR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_b5, b5, m_N); \
  deallocCudaDeviceData(b5); \
  deallocCudaDeviceData(stb5); \
  deallocCudaDeviceData(sa); \
  deallocCudaDeviceData(sb);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void genlinrecur1(Real_ptr b5, Real_ptr stb5,
                             Real_ptr sa, Real_ptr sb,
                             Index_type kb5i,
                             Index_type N)
{
   Index_type k = blockIdx.x * block_size + threadIdx.x;
   if (k < N) {
     GEN_LIN_RECUR_BODY1;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void genlinrecur2(Real_ptr b5, Real_ptr stb5,
                             Real_ptr sa, Real_ptr sb,
                             Index_type kb5i,
                             Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i > 0 && i < N+1) {
     GEN_LIN_RECUR_BODY2;
   }
}


template < size_t block_size >
void GEN_LIN_RECUR::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    GEN_LIN_RECUR_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(N, block_size);
       genlinrecur1<block_size><<<grid_size1, block_size>>>( b5, stb5, sa, sb,
                                                 kb5i,
                                                 N );
       cudaErrchk( cudaGetLastError() );

       const size_t grid_size2 = RAJA_DIVIDE_CEILING_INT(N+1, block_size);
       genlinrecur2<block_size><<<grid_size2, block_size>>>( b5, stb5, sa, sb,
                                                 kb5i,
                                                 N );
       cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    GEN_LIN_RECUR_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(0, N), [=] __device__ (Index_type k) {
         GEN_LIN_RECUR_BODY1;
       });

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(1, N+1), [=] __device__ (Index_type i) {
         GEN_LIN_RECUR_BODY2;
       });

    }
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  GEN_LIN_RECUR : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(GEN_LIN_RECUR, Cuda)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
