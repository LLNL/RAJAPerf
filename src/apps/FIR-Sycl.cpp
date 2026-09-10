//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define FIR_DATA_SETUP_SYCL \
  Real_ptr coeff; \
\
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitSyclDeviceData(coeff, tcoeff, FIR_COEFFLEN, qu);

#define FIR_DATA_TEARDOWN_SYCL \
  deallocSyclDeviceData(coeff, qu);


template <size_t work_group_size >
void FIR::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  FIR_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    FIR_COEFF;

    FIR_DATA_SETUP_SYCL;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            FIR_BODY
          }

        });
      });
      RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    FIR_COEFF;

    FIR_DATA_SETUP_SYCL;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("FIR_1");
       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         FIR_BODY;
       });
       RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  FIR : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIR, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
