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

#include <iostream>

namespace rajaperf
{
namespace basic
{

void DYANAMIC_TILE::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  DYANAMIC_TILE_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto body0 = [=](Index_type k, Index_type j, Index_type i) {
                 DYANAMIC_TILE_BODY(offset0, ni0, nj0);
               };
  auto body1 = [=](Index_type k, Index_type j, Index_type i) {
                 DYANAMIC_TILE_BODY(offset1, ni1, nj1);
               };
  auto body2 = [=](Index_type k, Index_type j, Index_type i) {
                 DYANAMIC_TILE_BODY(offset2, ni2, nj2);
               };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
        for (Index_type k = 0; k < nk0; ++k ) {
          for (Index_type j = 0; j < nj0; ++j ) {
            for (Index_type i = 0; i < ni0; ++i ) {
              DYANAMIC_TILE_BODY(offset0, ni0, nj0);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
        for (Index_type k = 0; k < nk1; ++k ) {
          for (Index_type j = 0; j < nj1; ++j ) {
            for (Index_type i = 0; i < ni1; ++i ) {
              DYANAMIC_TILE_BODY(offset1, ni1, nj1);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
        for (Index_type k = 0; k < nk2; ++k ) {
          for (Index_type j = 0; j < nj2; ++j ) {
            for (Index_type i = 0; i < ni2; ++i ) {
              DYANAMIC_TILE_BODY(offset2, ni2, nj2);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
        for (Index_type k = 0; k < nk0; ++k ) {
          for (Index_type j = 0; j < nj0; ++j ) {
            for (Index_type i = 0; i < ni0; ++i ) {
              body0(k, j, i);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
        for (Index_type k = 0; k < nk1; ++k ) {
          for (Index_type j = 0; j < nj1; ++j ) {
            for (Index_type i = 0; i < ni1; ++i ) {
              body1(k, j, i);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
        for (Index_type k = 0; k < nk2; ++k ) {
          for (Index_type j = 0; j < nj2; ++j ) {
            for (Index_type i = 0; i < ni2; ++i ) {
              body2(k, j, i);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      using EXEC_POL =
        RAJA::fornest_tiling_policy<RAJA::fornest_basic_seq_3d<RAJA::seq_exec>,
                                    RAJA::fornest_tile_auto,
                                    RAJA::fornest_tile_auto,
                                    RAJA::fornest_tile_auto>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
        RAJA::fornest(EXEC_POL {},
                      RAJA::range(nk0), RAJA::range(nj0), RAJA::range(ni0),
                      body0);
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
        RAJA::fornest(EXEC_POL {},
                      RAJA::range(nk1), RAJA::range(nj1), RAJA::range(ni1),
                      body1);
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

        RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
        RAJA::fornest(EXEC_POL {},
                      RAJA::range(nk2), RAJA::range(nj2), RAJA::range(ni2),
                      body2);
        RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  DYANAMIC_TILE : Unknown variant id = " << vid << std::endl;
    }

  }
}

void DYANAMIC_TILE::defineSeqVariantTunings()
{
  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {
    if (vid == RAJA_Seq) {
      addVariantTuning<&DYANAMIC_TILE::runSeqVariant>(
          vid, "fornest-auto-tile");
    } else {
      addVariantTuning<&DYANAMIC_TILE::runSeqVariant>(
          vid, getDefaultTuningName());
    }
  }
}

} // end namespace basic
} // end namespace rajaperf
