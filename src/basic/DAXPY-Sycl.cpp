//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;

#define DAXPY_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, iend, qu); \
  allocAndInitSyclDeviceData(y, m_y, iend, qu);

#define DAXPY_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_y, y, iend, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu);


void DAXPY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    DAXPY_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for(cl::sycl::nd_range<1>{grid_size, block_size},
                                    [=] (cl::sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            DAXPY_BODY
          }

        });
      });
    }

    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();
    DAXPY_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    DAXPY_DATA_SETUP_SYCL;
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<block_size, true> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        DAXPY_BODY;
      });


    }
    qu->wait();
    stopTimer();

    DAXPY_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  DAXPY : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
