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

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define IF_QUAD_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(a, m_a, iend, qu); \
  allocAndInitSyclDeviceData(b, m_b, iend, qu); \
  allocAndInitSyclDeviceData(c, m_c, iend, qu); \
  allocAndInitSyclDeviceData(x1, m_x1, iend, qu); \
  allocAndInitSyclDeviceData(x2, m_x2, iend, qu);

#define IF_QUAD_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_x1, x1, iend, qu); \
  getSyclDeviceData(m_x2, x2, iend, qu); \
  deallocSyclDeviceData(a, qu); \
  deallocSyclDeviceData(b, qu); \
  deallocSyclDeviceData(c, qu); \
  deallocSyclDeviceData(x1, qu); \
  deallocSyclDeviceData(x2, qu);

template <size_t work_group_size >
void IF_QUAD::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  IF_QUAD_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    if (work_group_size > 0) {

      IF_QUAD_DATA_SETUP_SYCL;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                         [=] (sycl::nd_item<1> item ) {
  
            Index_type i = item.get_global_id(0);
  
            if (i < iend) {
              IF_QUAD_BODY
            }
          });
        });
      }
      qu->wait(); // Wait for computation to finish before stopping timer
      stopTimer();
  
      IF_QUAD_DATA_TEARDOWN_SYCL;

    } else {

      IF_QUAD_DATA_SETUP_SYCL;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::range<1>(iend),
                         [=] (sycl::item<1> item) {
  
            Index_type i = item.get_id(0);
            IF_QUAD_BODY
  
          });
        });
      }
      qu->wait(); // Wait for computation to finish before stopping timer
      stopTimer();
  
      IF_QUAD_DATA_TEARDOWN_SYCL;

    }

  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  IF_QUAD : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    IF_QUAD_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<work_group_size, true> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         IF_QUAD_BODY;
       });

    }
    qu->wait();
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  IF_QUAD : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(IF_QUAD, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
