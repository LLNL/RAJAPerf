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

template <size_t work_group_size >
void DAXPY::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};

  DAXPY_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    if (work_group_size > 0) {

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

    } else {

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

    }

  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  DAXPY : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, true> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        DAXPY_BODY;
      });

    }
    qu->wait();
    stopTimer();

  } else {
     std::cout << "\n  DAXPY : Unknown Sycl variant id = " << vid << std::endl;
  }

}
RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DAXPY, Sycl)

} // end namespace basic
} // end namespace rajaperf


#endif  // RAJA_ENABLE_SYCL
