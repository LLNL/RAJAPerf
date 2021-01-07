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

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(a, m_a, iend, qu);

#define INIT_VIEW1D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_a, a, iend, qu); \
  deallocSyclDeviceData(a, qu);

void INIT_VIEW1D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    {
      INIT_VIEW1D_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {

          const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

          h.parallel_for<class syclInit3_view1d>(cl::sycl::nd_range<1>{grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item ) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              INIT_VIEW1D_BODY
            }
          });
        });
        qu.wait();
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    INIT_VIEW1D_DATA_TEARDOWN_SYCL;
/*
  } else if ( vid == RAJA_SYCL ) {

    INIT_VIEW1D_DATA_SETUP_SYCL;

    INIT_VIEW1D_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<block_size  /*async*//*> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        INIT_VIEW1D_BODY_RAJA;
      });

    }
    qu.wait();
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_SYCL;
*/
  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
