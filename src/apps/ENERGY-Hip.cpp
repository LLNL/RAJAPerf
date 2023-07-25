//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc1(Real_ptr e_new, Real_ptr e_old, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old, Real_ptr work,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY1;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc2(Real_ptr delvc, Real_ptr q_new,
                            Real_ptr compHalfStep, Real_ptr pHalfStep,
                            Real_ptr e_new, Real_ptr bvc, Real_ptr pbvc,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY2;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc3(Real_ptr e_new, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY3;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc4(Real_ptr e_new, Real_ptr work,
                            Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY4;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc5(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Real_type rho0, Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY5;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void energycalc6(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr q_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0, Real_type q_cut,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY6;
   }
}


template < size_t block_size >
void ENERGY::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ENERGY_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       constexpr size_t shmem = 0;

       hipLaunchKernelGGL((energycalc1<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  e_new, e_old, delvc,
                                               p_old, q_old, work,
                                               iend );
       hipErrchk( hipGetLastError() );

       hipLaunchKernelGGL((energycalc2<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  delvc, q_new,
                                               compHalfStep, pHalfStep,
                                               e_new, bvc, pbvc,
                                               ql_old, qq_old,
                                               rho0,
                                               iend );
       hipErrchk( hipGetLastError() );

       hipLaunchKernelGGL((energycalc3<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  e_new, delvc,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               iend );
       hipErrchk( hipGetLastError() );

       hipLaunchKernelGGL((energycalc4<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  e_new, work,
                                               e_cut, emin,
                                               iend );
       hipErrchk( hipGetLastError() );

       hipLaunchKernelGGL((energycalc5<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               ql_old, qq_old,
                                               p_old, q_old,
                                               pHalfStep, q_new,
                                               rho0, e_cut, emin,
                                               iend );
       hipErrchk( hipGetLastError() );

       hipLaunchKernelGGL((energycalc6<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  delvc,
                                               pbvc, e_new, vnewc,
                                               bvc, p_new,
                                               q_new,
                                               ql_old, qq_old,
                                               rho0, q_cut,
                                               iend );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY4;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ENERGY_BODY6;
        });

      });  // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     getCout() << "\n  ENERGY : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ENERGY, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
