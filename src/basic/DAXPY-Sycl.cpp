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

#define DAXPY_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, iend, qu); \
  allocAndInitSyclDeviceData(y, m_y, iend, qu);

#define DAXPY_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_y, y, iend, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu);

template <size_t work_group_size >
void DAXPY::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    if (work_group_size > 0) {

      DAXPY_DATA_SETUP_SYCL;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1>{global_size, work_group_size},
                         [=] (sycl::nd_item<1> item ) {
  
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
    } else {

      DAXPY_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::range<1>(iend),
                         [=] (sycl::item<1> item) {

            Index_type i = item.get_id(0);
            DAXPY_BODY

          });
        });
      }
      qu->wait(); // Wait for computation to finish before stopping timer

      stopTimer();

      DAXPY_DATA_TEARDOWN_SYCL;
    }

  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  INIT3 : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    DAXPY_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, false> >(
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

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DAXPY, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
