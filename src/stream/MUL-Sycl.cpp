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

#include "MUL.hpp"

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

#define MUL_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(b, m_b, iend, qu); \
  allocAndInitSyclDeviceData(c, m_c, iend, qu);

#define MUL_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_b, b, iend, qu); \
  deallocSyclDeviceData(b, qu); \
  deallocSyclDeviceData(c, qu)

void MUL::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  MUL_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    MUL_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class MUL>(cl::sycl::nd_range<1> (grid_size, block_size),
                                  [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            MUL_BODY
          }
        });
      });
    }

    qu->wait();
    stopTimer();

    MUL_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    MUL_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         MUL_BODY;
       });

    }
    qu->wait();
    stopTimer();

    MUL_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  MUL : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
