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

#include "NESTED_INIT.hpp"

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
  const size_t block_size = 256;

#define NESTED_INIT_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(array, m_array, m_array_length, qu);

#define NESTED_INIT_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_array, array, m_array_length, qu); \
  deallocSyclDeviceData(array, qu);

void NESTED_INIT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    NESTED_INIT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(ni, block_size); 

      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class NestedInit>(cl::sycl::nd_range<3> (
                                             cl::sycl::range<3> (nk, nj, grid_size),
                                             cl::sycl::range<3> (1, 1, block_size)),
                                         [=] (cl::sycl::nd_item<3> item) {

          Index_type i = item.get_global_id(2);
          Index_type j = item.get_global_id(1);
          Index_type k = item.get_global_id(0);

          if (i < ni) {
            NESTED_INIT_BODY
          }
        });
      });
    }
    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    NESTED_INIT_DATA_SETUP_SYCL;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<2, RAJA::sycl_global_2<1>,      // k
            RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
              RAJA::statement::For<0, RAJA::sycl_global_0<256>, // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
        [=] (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    qu->wait();
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
