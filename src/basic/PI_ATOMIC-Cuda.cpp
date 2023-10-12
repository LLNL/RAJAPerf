//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pi_atomic(Real_ptr pi,
                          Real_type dx,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     double x = (double(i) + 0.5) * dx;
     RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
   }
}



template < size_t block_size >
void PI_ATOMIC::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_ATOMIC_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync( pi, &m_pi_init, sizeof(Real_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      pi_atomic<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>( pi, dx, iend );
      cudaErrchk( cudaGetLastError() );

      cudaErrchk( cudaMemcpyAsync( &m_pi_final, pi, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_pi_final *= 4.0;

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync( pi, &m_pi_init, sizeof(Real_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
      });
      cudaErrchk( cudaGetLastError() );

      cudaErrchk( cudaMemcpyAsync( &m_pi_final, pi, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_pi_final *= 4.0;

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaMemcpyAsync( pi, &m_pi_init, sizeof(Real_type),
                                   cudaMemcpyHostToDevice, res.get_stream() ) );

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::cuda_atomic>(pi, dx / (1.0 + x * x));
      });

      cudaErrchk( cudaMemcpyAsync( &m_pi_final, pi, sizeof(Real_type),
                                   cudaMemcpyDeviceToHost, res.get_stream() ) );
      cudaErrchk( cudaStreamSynchronize( res.get_stream() ) );
      m_pi_final *= 4.0;

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PI_ATOMIC, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
