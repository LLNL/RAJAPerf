//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MATVEC_3D_STENCIL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace apps
{

template <size_t work_group_size >
void MATVEC_3D_STENCIL::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MATVEC_3D_STENCIL_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MATVEC_3D_STENCIL_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type ii = item.get_global_id(0);
          Index_type i = ii + ibegin;
          if (i < iend) {
            MATVEC_3D_STENCIL_BODY_INDEX;
            MATVEC_3D_STENCIL_BODY;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("MATVEC_3D_STENCIL_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MATVEC_3D_STENCIL_1");
      RAJA::forall<RAJA::sycl_exec<work_group_size, true /*async*/>>(res,
          zones, [=](Index_type i) {
        MATVEC_3D_STENCIL_BODY;
      });
      RP_CALI_SUBKERNEL_END("MATVEC_3D_STENCIL_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  MATVEC_3D_STENCIL : Unknown Sycl variant id = " << vid << std::endl;
  }

}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MATVEC_3D_STENCIL, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
