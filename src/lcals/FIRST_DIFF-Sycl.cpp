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

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define FIRST_DIFF_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, m_N, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_N, qu);

#define FIRST_DIFF_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_x, x, m_N, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu);

void FIRST_DIFF::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  FIRST_DIFF_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    FIRST_DIFF_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);
      qu.submit([&] (cl::sycl::handler& h)
      {
        h.parallel_for<class syclFirstDiff>(cl::sycl::nd_range<1>(grid_size, block_size),
                                            [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            FIRST_DIFF_BODY
          }

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    FIRST_DIFF_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         FIRST_DIFF_BODY;
       });

    }
    qu.wait();
    stopTimer();

    FIRST_DIFF_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  FIRST_DIFF : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
