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

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define INIT3_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(out1, m_out1, iend, qu); \
  allocAndInitSyclDeviceData(out2, m_out2, iend, qu); \
  allocAndInitSyclDeviceData(out3, m_out3, iend, qu); \
  allocAndInitSyclDeviceData(in1, m_in1, iend, qu); \
  allocAndInitSyclDeviceData(in2, m_in2, iend, qu);

#define INIT3_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_out1, out1, iend, qu); \
  getSyclDeviceData(m_out2, out2, iend, qu); \
  getSyclDeviceData(m_out3, out3, iend, qu); \
  deallocSyclDeviceData(out1, qu); \
  deallocSyclDeviceData(out2, qu); \
  deallocSyclDeviceData(out3, qu); \
  deallocSyclDeviceData(in1, qu); \
  deallocSyclDeviceData(in2, qu);

template <size_t work_group_size >
void INIT3::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT3_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    if (work_group_size > 0) {
  
      INIT3_DATA_SETUP_SYCL;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                                           [=] (sycl::nd_item<1> item ) {
  
            Index_type i = item.get_global_id(0);
            if (i < iend) {
              INIT3_BODY
            }
  
          });
        });
  
      }
      qu->wait();
  
      stopTimer();
  
      INIT3_DATA_TEARDOWN_SYCL;
  
    } else {
  
      INIT3_DATA_SETUP_SYCL;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::range<1>(iend),
                                        [=] (sycl::item<1> item ) {
  
            Index_type i = item.get_id(0);
            INIT3_BODY
  
          });
        });
  
      }
      qu->wait();
  
      stopTimer();
  
      INIT3_DATA_TEARDOWN_SYCL;

    } 
  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  INIT3 : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    INIT3_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        INIT3_BODY;
      });

    }
    qu->wait();
    stopTimer();

    INIT3_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  INIT3 : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INIT3, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
