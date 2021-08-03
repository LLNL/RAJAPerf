//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define REDUCE3_INT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(vec);


__global__ void emptykernel597679()
{
}


void REDUCE3_INT::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    Int_ptr vmem_init;
    allocHipPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocHipDeviceData(vmem, 3);

    std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3, t4;
    t0 = std::chrono::high_resolution_clock::now();

    REDUCE3_INT_DATA_SETUP_HIP;

    t2 = std::chrono::high_resolution_clock::now();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      hipErrchk( hipMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                 hipMemcpyHostToDevice, camp::resources::Hip::get_default().get_stream() ) );

      t3 = std::chrono::high_resolution_clock::now();

      auto func = (const void*)(emptykernel597679);
      void* args[] = {nullptr};
      hipErrchk( hipLaunchKernel( func, 1, 1, args, 0, camp::resources::Hip::get_default().get_stream() ) );

    }
    stopTimer();

    t4 = std::chrono::high_resolution_clock::now();

    double us2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
    double us3 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    double us4 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::printf("%s took %.3fus %.3fus %.3fus\n", "REDUCE3_INT_Base_HIP", us2, us3, us4); std::fflush(stdout);

    REDUCE3_INT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(vmem);
    deallocHipPinnedData(vmem_init);

  } else if ( vid == RAJA_HIP ) {

    std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1, t2, t3, t4;
    t0 = std::chrono::high_resolution_clock::now();

    REDUCE3_INT_DATA_SETUP_HIP;

    t2 = std::chrono::high_resolution_clock::now();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      t3 = std::chrono::high_resolution_clock::now();

      auto func = (const void*)(emptykernel597679);
      void* args[] = {nullptr};
      hipErrchk( hipLaunchKernel( func, 1, 1, args, 0, camp::resources::Hip::get_default().get_stream() ) );

    }
    stopTimer();

    t4 = std::chrono::high_resolution_clock::now();

    double us2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
    double us3 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    double us4 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    std::printf("%s took %.3fus %.3fus %.3fus\n", "REDUCE3_INT_RAJA_HIP", us2, us3, us4); std::fflush(stdout);

    REDUCE3_INT_DATA_TEARDOWN_HIP;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
