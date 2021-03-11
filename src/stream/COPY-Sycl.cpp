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

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace stream
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;

#define COPY_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(a, m_a, iend, qu); \
  allocAndInitSyclDeviceData(c, m_c, iend, qu);

#define COPY_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_c, c, iend, qu); \
  deallocSyclDeviceData(a, qu); \
  deallocSyclDeviceData(c, qu);

void COPY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  COPY_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    COPY_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class COPY>(cl::sycl::nd_range<1> (grid_size, block_size),
                                   [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            COPY_BODY
          }

        });
      });
    }

    qu.wait();
    stopTimer();

    COPY_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    COPY_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         COPY_BODY;
       });

    }
    qu.wait();
    stopTimer();

    COPY_DATA_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  COPY : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
