//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
    PI_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_CUDA);
  }
}



template < size_t block_size >
void PI_ATOMIC::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_ATOMIC_GPU_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, pi, hpi, 1, 1);

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (pi_atomic<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          pi,
                          dx, 
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(pi, hpi, 1, 1);
      m_pi_final = hpi[0] * static_cast<Real_type>(4);
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1, 1);

      auto pi_atomic_lambda = [=] __device__ (Index_type i) {
        PI_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_CUDA);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(pi_atomic_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, pi_atomic_lambda );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(pi, hpi, 1, 1);
      m_pi_final = hpi[0] * static_cast<Real_type>(4);
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PI_ATOMIC_1");
      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1, 1);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PI_ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_CUDA);
      });

      RAJAPERF_CUDA_REDUCER_COPY_BACK(pi, hpi, 1, 1);
      m_pi_final = hpi[0] * static_cast<Real_type>(4);
      RP_CALI_SUBKERNEL_END("PI_ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(pi, hpi);

}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PI_ATOMIC, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
