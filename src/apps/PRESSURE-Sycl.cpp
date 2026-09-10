//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

template <size_t work_group_size >
void PRESSURE::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  PRESSURE_DATA_SETUP;

  using sycl::fabs;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY1
          }

        });
      });
      RP_CALI_SUBKERNEL_END("PRESSURE_1");

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PRESSURE_BODY2
          }

        });
      });
      RP_CALI_SUBKERNEL_END("PRESSURE_2");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    const bool async = true;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          PRESSURE_BODY1;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_1");

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          PRESSURE_BODY2;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_2");

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     std::cout << "\n  PRESSURE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PRESSURE, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
