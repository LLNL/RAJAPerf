//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>
#include <cmath>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

template <size_t work_group_size >
void PLANCKIAN::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue(); 

  PLANCKIAN_DATA_SETUP;

  using sycl::exp;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PLANCKIAN_BODY;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("PLANCKIAN_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("PLANCKIAN_1");
       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         PLANCKIAN_BODY;
       });
       RP_CALI_SUBKERNEL_END("PLANCKIAN_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  PLANCKIAN : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PLANCKIAN, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
