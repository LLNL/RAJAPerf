//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

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
void GEN_LIN_RECUR::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       constexpr size_t shmem = 0;

       const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(N, block_size);
       hipLaunchKernelGGL((genlinrecur1<block_size>), grid_size1, block_size, shmem, res.get_stream(),
                                                 b5, stb5, sa, sb,
                                                 kb5i,
                                                 N );
       hipErrchk( hipGetLastError() );

       const size_t grid_size2 = RAJA_DIVIDE_CEILING_INT(N+1, block_size);
       hipLaunchKernelGGL((genlinrecur2<block_size>), grid_size2, block_size, shmem, res.get_stream(),
                                                 b5, stb5, sa, sb,
                                                 kb5i,
                                                 N );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(0, N), [=] __device__ (Index_type k) {
         GEN_LIN_RECUR_BODY1;
       });

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(1, N+1), [=] __device__ (Index_type i) {
         GEN_LIN_RECUR_BODY2;
       });

    }
    stopTimer();

  } else {
     getCout() << "\n  GEN_LIN_RECUR : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, Hip)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
