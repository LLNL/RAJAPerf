//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  //  const size_t block_size = 256;

  /*
#define MAT_MAT_SHARED_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(out1, m_out1, iend); \
  allocAndInitCudaDeviceData(out2, m_out2, iend); \
  allocAndInitCudaDeviceData(out3, m_out3, iend); \
  allocAndInitCudaDeviceData(in1, m_in1, iend); \
  allocAndInitCudaDeviceData(in2, m_in2, iend);

#define MAT_MAT_SHARED_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out1, out1, iend); \
  getCudaDeviceData(m_out2, out2, iend); \
  getCudaDeviceData(m_out3, out3, iend); \
  deallocCudaDeviceData(out1); \
  deallocCudaDeviceData(out2); \
  deallocCudaDeviceData(out3); \
  deallocCudaDeviceData(in1); \
  deallocCudaDeviceData(in2);
  */
__global__ void mat_mat_shared(Real_ptr out1, Real_ptr out2, Real_ptr out3,
                               Real_ptr in1, Real_ptr in2,
                               Index_type iend)
{
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    //MAT_MAT_SHARED_BODY;
  }
}


void MAT_MAT_SHARED::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  //MAT_MAT_SHARED_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    //MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      //muladdsub<<<grid_size, block_size>>>( out1, out2, out3, in1, in2,
      //iend );

    }
    stopTimer();

    //MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    //MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      /*
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        MAT_MAT_SHARED_BODY;
      });
      */

    }
    stopTimer();

    //MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    //MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #if 0
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MAT_MAT_SHARED_BODY;
      });
      #endif

    }
    stopTimer();

    //MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  MAT_MAT_SHARED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
