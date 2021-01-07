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

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 128;


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

void IF_QUAD::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  IF_QUAD_DATA_SETUP;
  using cl::sycl::sqrt;

  if ( vid == Base_SYCL ) {

    IF_QUAD_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class IfQuad>(cl::sycl::nd_range<1>(grid_size, block_size),
                                     [=] (cl::sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);

          if (i < iend) {
            IF_QUAD_BODY
          }
        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    IF_QUAD_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
       //  using cl::sycl::sqrt;
         IF_QUAD_BODY;
       });

    }
    qu.wait();
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  IF_QUAD : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
