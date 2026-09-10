//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace stream
{

template <size_t work_group_size >
void TRIAD::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  TRIAD_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRIAD_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            TRIAD_BODY
          }

        });
      });
      RP_CALI_SUBKERNEL_END("TRIAD_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("TRIAD_1");
       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         TRIAD_BODY;
       });
       RP_CALI_SUBKERNEL_END("TRIAD_1");

    }
    stopTimer();

  } else {
      std::cout << "\n  TRIAD : Unknown Sycl variant id = " << vid << std::endl;
  }

}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIAD, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
