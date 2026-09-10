//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void int_predict(Real_ptr px,
                            Real_type dm22, Real_type dm23, Real_type dm24,
                            Real_type dm25, Real_type dm26, Real_type dm27,
                            Real_type dm28, Real_type c0,
                            const Index_type offset,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     INT_PREDICT_BODY;
   }
}


template < size_t block_size >
void INT_PREDICT::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  INT_PREDICT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("INT_PREDICT_1");
       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       constexpr size_t shmem = 0;
      
       RPlaunchCudaKernel( (int_predict<block_size>),
                           grid_size, block_size,
                           shmem, res.get_stream(),
                           px,
                           dm22, dm23, dm24,
                           dm25, dm26, dm27,
                           dm28, c0,
                           offset,
                           iend );
       RP_CALI_SUBKERNEL_END("INT_PREDICT_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("INT_PREDICT_1");
       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         INT_PREDICT_BODY;
       });
       RP_CALI_SUBKERNEL_END("INT_PREDICT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  INT_PREDICT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INT_PREDICT, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
