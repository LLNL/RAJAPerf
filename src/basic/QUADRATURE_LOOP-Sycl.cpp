//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "QUADRATURE_LOOP.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t work_group_size >
void QUADRATURE_LOOP::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  QUADRATURE_LOOP_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("QUADRATURE_LOOP_1");
      const size_t global_size =
        work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                                        [=] (sycl::nd_item<1> item ) {

          Index_type idx = item.get_global_id(0);
          if (idx < iend) {
            Index_type zone = idx / 27;
            Index_type q = idx - 27 * zone;
            QUADRATURE_LOOP_BODY;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("QUADRATURE_LOOP_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    using FORNEST_POL =
      RAJA::fornest_collapsed_policy<RAJA::sycl_exec<work_group_size>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("QUADRATURE_LOOP_1");
      RAJA::fornest(res, FORNEST_POL {},
                    RAJA::range(num_zones), RAJA::range(27),
                    [=] (Index_type zone, Index_type q) {
        QUADRATURE_LOOP_BODY;
      });
      RP_CALI_SUBKERNEL_END("QUADRATURE_LOOP_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  QUADRATURE_LOOP : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(QUADRATURE_LOOP, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
