//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DYANAMIC_TILE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t work_group_size >
void DYANAMIC_TILE::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  DYANAMIC_TILE_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    auto submit = [&](Index_type offset,
                      Index_type ni, Index_type nj, Index_type nk) {
      const Index_type len = ni * nj * nk;
      const size_t global_size =
        work_group_size * RAJA_DIVIDE_CEILING_INT(len, work_group_size);

      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {
          Index_type flat = item.get_global_id(0);
          if (flat < len) {
            Index_type i = flat % ni;
            Index_type j = (flat / ni) % nj;
            Index_type k = flat / (ni * nj);
            DYANAMIC_TILE_BODY(offset, ni, nj);
          }
        });
      });
    };

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
      submit(offset0, ni0, nj0, nk0);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
      submit(offset1, ni1, nj1, nk1);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
      submit(offset2, ni2, nj2, nk2);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::sycl_exec<work_group_size>,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto>;

    auto body0 = [=] (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset0, ni0, nj0);
    };
    auto body1 = [=] (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset1, ni1, nj1);
    };
    auto body2 = [=] (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset2, ni2, nj2);
    };

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk0), RAJA::range(nj0), RAJA::range(ni0),
                    body0);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk1), RAJA::range(nj1), RAJA::range(ni1),
                    body1);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk2), RAJA::range(nj2), RAJA::range(ni2),
                    body2);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

    }
    stopTimer();

  } else {
     getCout() << "\n  DYANAMIC_TILE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

void DYANAMIC_TILE::defineSyclVariantTunings()
{
  for (VariantID vid : {Base_SYCL, RAJA_SYCL}) {

    seq_for(gpu_block_sizes_type{}, [&](auto work_group_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(work_group_size)) {

        if (vid == RAJA_SYCL) {
          addVariantTuning<&DYANAMIC_TILE::runSyclVariantImpl<work_group_size>>(
              vid, "fornest-auto-tile_block_"+std::to_string(work_group_size));
        } else {
          addVariantTuning<&DYANAMIC_TILE::runSyclVariantImpl<work_group_size>>(
              vid, "block_"+std::to_string(work_group_size));
        }

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
