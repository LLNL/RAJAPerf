//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define REDUCE3_INT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(vec);

#define REDUCE3_INT_BODY_HIP(atomicAdd, atomicMin, atomicMax) \
  RAJAPERF_REDUCE_3_HIP(Int_type, REDUCE3_INT_VALS, vsum, vsum_init, RAJAPERF_ADD_OP, atomicAdd, \
                                                    vmin, vmin_init, RAJAPERF_MIN_OP, atomicMin, \
                                                    vmax, vmax_init, RAJAPERF_MAX_OP, atomicMax)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_type vsum_init,
                           Int_ptr vmin, Int_type vmin_init,
                           Int_ptr vmax, Int_type vmax_init,
                           Index_type iend)
{
  REDUCE3_INT_BODY_HIP(::atomicAdd,
                       ::atomicMin,
                       ::atomicMax)
}



template < size_t block_size >
void REDUCE3_INT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE3_INT_DATA_SETUP_HIP;

    Int_ptr vmem_init;
    allocHipPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocHipDeviceData(vmem, 3);
    Int_ptr vsum = vmem + 0;
    Int_ptr vmin = vmem + 1;
    Int_ptr vmax = vmem + 2;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = vsum_init;
      vmem_init[1] = vmin_init;
      vmem_init[2] = vmax_init;
      hipErrchk( hipMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                 hipMemcpyHostToDevice ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((reduce3int<block_size>), dim3(grid_size), dim3(block_size), 3*sizeof(Int_type)*block_size, 0,
                                                    vec,
                                                    vsum, vsum_init,
                                                    vmin, vmin_init,
                                                    vmax, vmax_init,
                                                    iend );
      hipErrchk( hipGetLastError() );

      Int_type lmem[3];
      Int_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, vmem, 3);
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(vmem);
    deallocHipPinnedData(vmem_init);

  } else if ( vid == Lambda_HIP ) {

    REDUCE3_INT_DATA_SETUP_HIP;

    Int_ptr vmem_init;
    allocHipPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocHipDeviceData(vmem, 3);
    Int_ptr vsum = vmem + 0;
    Int_ptr vmin = vmem + 1;
    Int_ptr vmax = vmem + 2;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = vsum_init;
      vmem_init[1] = vmin_init;
      vmem_init[2] = vmax_init;
      hipErrchk( hipMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                 hipMemcpyHostToDevice ) );

      auto reduce3int_lambda = [=] __device__ () {
        REDUCE3_INT_BODY_HIP(::atomicAdd,
                             ::atomicMin,
                             ::atomicMax)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip<block_size, decltype(reduce3int_lambda)>),
                         dim3(grid_size), dim3(block_size), 3*sizeof(Int_type)*block_size, 0,
                         reduce3int_lambda );
      hipErrchk( hipGetLastError() );

      Int_type lmem[3];
      Int_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, vmem, 3);
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(vmem);
    deallocHipPinnedData(vmem_init);

  } else if ( vid == RAJA_HIP ) {

    REDUCE3_INT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Int_type> vsum(vsum_init);
      RAJA::ReduceMin<RAJA::hip_reduce, Int_type> vmin(vmin_init);
      RAJA::ReduceMax<RAJA::hip_reduce, Int_type> vmax(vmax_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(REDUCE3_INT, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
