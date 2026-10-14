//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void NESTED_INIT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
        for (Index_type k = 0; k < nk; ++k ) {
          for (Index_type j = 0; j < nj; ++j ) {
            for (Index_type i = 0; i < ni; ++i ) {
              NESTED_INIT_BODY;
            }
          }
        }
        RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                nestedinit_lam(i, j, k);
              }
            }
          }
          RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<2, RAJA::seq_exec,    // k
            RAJA::statement::For<1, RAJA::seq_exec,  // j
              RAJA::statement::For<0, RAJA::seq_exec,// i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
        RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                          RAJA::RangeSegment(0, nj),
                                                          RAJA::RangeSegment(0, nk)),
                                         res,
                                         nestedinit_lam
                                       );
        RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void NESTED_INIT::runSeqVariantFornest(VariantID vid)
{
#if defined(RUN_RAJA_SEQ)
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type k, Index_type j, Index_type i) {
                          NESTED_INIT_BODY;
                        };

  if ( vid == RAJA_Seq ) {

    using EXEC_POL = RAJA::fornest_basic_seq_3d<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    nestedinit_lam);
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

template < size_t tile_k, size_t tile_j, size_t tile_i >
void NESTED_INIT::runSeqVariantFornestRuntimeTiled(VariantID vid)
{
#if defined(RUN_RAJA_SEQ)
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type k, Index_type j, Index_type i) {
                          NESTED_INIT_BODY;
                        };

  if ( vid == RAJA_Seq ) {

    using BASE_POL = RAJA::fornest_basic_seq_3d<RAJA::seq_exec>;
    using EXEC_POL =
      RAJA::fornest_tiling_policy<BASE_POL,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(EXEC_POL {RAJA::TileSize(tile_k),
                              RAJA::TileSize(tile_j),
                              RAJA::TileSize(tile_i)},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    nestedinit_lam);
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void NESTED_INIT::runSeqVariantFornestAutoTiled(VariantID vid)
{
#if defined(RUN_RAJA_SEQ)
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type k, Index_type j, Index_type i) {
                          NESTED_INIT_BODY;
                        };

  if ( vid == RAJA_Seq ) {

    using BASE_POL = RAJA::fornest_basic_seq_3d<RAJA::seq_exec>;
    using EXEC_POL =
      RAJA::fornest_tiling_policy<BASE_POL,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    nestedinit_lam);
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void NESTED_INIT::defineSeqVariantTunings()
{
  for (VariantID vid : {Base_Seq, Lambda_Seq, RAJA_Seq}) {

    addVariantTuning<&NESTED_INIT::runSeqVariant>(
        vid, getDefaultTuningName());

    if (vid == RAJA_Seq) {
      addVariantTuning<&NESTED_INIT::runSeqVariantFornest>(
          vid, "fornest");
      addVariantTuning<&NESTED_INIT::runSeqVariantFornestRuntimeTiled<1, 8, 32>>(
          vid, "fornest-runtime-tile_1x8x32");
      addVariantTuning<&NESTED_INIT::runSeqVariantFornestRuntimeTiled<2, 4, 32>>(
          vid, "fornest-runtime-tile_2x4x32");
      addVariantTuning<&NESTED_INIT::runSeqVariantFornestRuntimeTiled<4, 4, 16>>(
          vid, "fornest-runtime-tile_4x4x16");
      addVariantTuning<&NESTED_INIT::runSeqVariantFornestAutoTiled>(
          vid, "fornest-auto-tile");
    }

  }
}

} // end namespace basic
} // end namespace rajaperf
