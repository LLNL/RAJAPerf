//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define TRIDIAG_ELIM_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(xout, m_xout, m_N, qu); \
  allocAndInitSyclDeviceData(xin, m_xin, m_N, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_N, qu); \
  allocAndInitSyclDeviceData(z, m_z, m_N, qu);

#define TRIDIAG_ELIM_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_xout, xout, m_N, qu); \
  deallocSyclDeviceData(xout, qu); \
  deallocSyclDeviceData(xin, qu); \
  deallocSyclDeviceData(y, qu); \
  deallocSyclDeviceData(z, qu);

void TRIDIAG_ELIM::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    TRIDIAG_ELIM_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);
      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class TridiagElim>(cl::sycl::nd_range<1>(grid_size, block_size),
                                          [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i > 0 && i < iend) {
            TRIDIAG_ELIM_BODY;
          }

        });
      });
    }
    qu->wait();
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    TRIDIAG_ELIM_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         TRIDIAG_ELIM_BODY;
       });

    }
    qu->wait();
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  TRIDIAG_ELIM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
