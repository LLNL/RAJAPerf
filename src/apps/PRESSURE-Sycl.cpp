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

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define PRESSURE_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(compression, m_compression, iend, qu); \
  allocAndInitSyclDeviceData(bvc, m_bvc, iend, qu); \
  allocAndInitSyclDeviceData(p_new, m_p_new, iend, qu); \
  allocAndInitSyclDeviceData(e_old, m_e_old, iend, qu); \
  allocAndInitSyclDeviceData(vnewc, m_vnewc, iend, qu);

#define PRESSURE_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_p_new, p_new, iend, qu); \
  deallocSyclDeviceData(compression, qu); \
  deallocSyclDeviceData(bvc, qu); \
  deallocSyclDeviceData(p_new, qu); \
  deallocSyclDeviceData(e_old, qu); \
  deallocSyclDeviceData(vnewc, qu);

void PRESSURE::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  PRESSURE_DATA_SETUP;
  using cl::sycl::fabs;

  if ( vid == Base_SYCL ) {

    PRESSURE_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class PRESSURE_1>(cl::sycl::nd_range<1> (grid_size, block_size),
                                         [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY1
          }

        });
      });

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class PRESSURE_2>(cl::sycl::nd_range<1> (grid_size, block_size),
                                        [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY2
          }

        });
      });

    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    PRESSURE_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    PRESSURE_DATA_SETUP_SYCL;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          PRESSURE_BODY1;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          PRESSURE_BODY2;
        });

      }); // end sequential region (for single-source code)

    }
    qu.wait();
    stopTimer();

    PRESSURE_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  PRESSURE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
