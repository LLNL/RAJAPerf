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

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define INIT_VIEW1D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(a, m_a, iend, qu);

#define INIT_VIEW1D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_a, a, iend, qu); \
  deallocSyclDeviceData(a, qu);

template <size_t work_group_size >
void INIT_VIEW1D::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    INIT_VIEW1D_DATA_SETUP_SYCL;

    if (work_group_size > 0) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

        qu->submit([&] (sycl::handler& h) {

          h.parallel_for(sycl::nd_range<1>{global_size, work_group_size},
                                          [=] (sycl::nd_item<1> item ) {

            Index_type i = item.get_global_id(0);
            if (i < iend) {
              INIT_VIEW1D_BODY
            }
          });
        });
      }
      qu->wait();
      stopTimer();

    } else {
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::range<1>(iend),
                         [=] (sycl::item<1> item) {
  
            Index_type i = item.get_id(0);
            INIT_VIEW1D_BODY
  
          });
        });
      }
      qu->wait();
      stopTimer();

    }

    INIT_VIEW1D_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  INIT3 : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    INIT_VIEW1D_DATA_SETUP_SYCL;

    INIT_VIEW1D_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<work_group_size  /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        INIT_VIEW1D_BODY_RAJA;
      });

    }
    qu->wait();
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INIT_VIEW1D, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL