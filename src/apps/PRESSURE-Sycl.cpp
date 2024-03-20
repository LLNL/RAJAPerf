//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <sycl.hpp>
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
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;
  using sycl::fabs;

  if ( vid == Base_SYCL ) {

    PRESSURE_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for<class PRESSURE_1>(sycl::nd_range<1> (grid_size, block_size),
                                         [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY1
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for<class PRESSURE_2>(sycl::nd_range<1> (grid_size, block_size),
                                        [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY2
          }

        });
      });

    }
    qu->wait(); // Wait for computation to finish before stopping timer
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
    qu->wait();
    stopTimer();

    PRESSURE_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  PRESSURE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
