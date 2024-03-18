//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
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

#define FIR_DATA_SETUP_SYCL \
  Real_ptr coeff; \
\
  allocAndInitSyclDeviceData(in, m_in, getActualProblemSize(), qu); \
  allocAndInitSyclDeviceData(out, m_out, getActualProblemSize(), qu); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitSyclDeviceData(coeff, tcoeff, FIR_COEFFLEN, qu);


#define FIR_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_out, out, getActualProblemSize(), qu); \
  deallocSyclDeviceData(in, qu); \
  deallocSyclDeviceData(out, qu); \
  deallocSyclDeviceData(coeff, qu);

void FIR::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize() - m_coefflen;

  FIR_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    FIR_COEFF;

    FIR_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for<class Fir>(sycl::nd_range<1> (grid_size, block_size),
                                  [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            FIR_BODY
          }

        });
      });
    }
    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    FIR_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    FIR_COEFF;

    FIR_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         FIR_BODY;
       });

    }
    qu->wait();
    stopTimer();

    FIR_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  FIR : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
