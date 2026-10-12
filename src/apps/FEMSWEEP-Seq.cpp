//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FEMSWEEP.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


template < bool unroll >
void FEMSWEEP::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  FEMSWEEP_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
        for (Index_type a = 0; a < na; ++a)
        {
          for (Index_type g = 0; g < ng; ++g)
          {
            FEMSWEEP_KERNEL_SETUP;
            Index_type nehp_pos = 0;
            for (Index_type hp = 0; hp < nhp; ++hp)
            {
              const Index_type nehp = phpaa_r[ohp + hp];
              for (Index_type k = 0; k < nehp; ++k)
              {
                femsweepHyperplaneElement<unroll>(
                    Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
                    ne, ng, sharedinteriorfaces, order_r,
                    AngleElem2FaceType, elem_to_faces, F_g2l, idx1, idx2,
                    a, g, k, nehp_pos, Ffactor);
              }
              nehp_pos += nehp;
            }
          }
        }
        RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      using launch_policy =
          RAJA::LaunchPolicy<RAJA::seq_launch_t>;

      using outer_x =
          RAJA::LoopPolicy<RAJA::seq_exec>;

      using outer_y =
          RAJA::LoopPolicy<RAJA::seq_exec>;

      using inner_x =
          RAJA::LoopPolicy<RAJA::seq_exec>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("FEMSWEEP_1");
        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(),
            [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer_y>(ctx, RAJA::RangeSegment(0, na),
              [&](Index_type a) {
            RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, ng),
                [&](Index_type g) {
               FEMSWEEP_KERNEL_SETUP;
               Index_type nehp_pos = 0;
               for (Index_type hp = 0; hp < nhp; ++hp)
               {
                 const Index_type nehp = phpaa_r[ohp + hp];
                 RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, nehp),
                     [&](Index_type k) {
                   femsweepHyperplaneElement<unroll>(
                       Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
                       ne, ng, sharedinteriorfaces, order_r,
                       AngleElem2FaceType, elem_to_faces, F_g2l, idx1, idx2,
                       a, g, k, nehp_pos, Ffactor);
                 });  // k loop
                 ctx.teamSync();
                 nehp_pos += nehp;
               }
             });  // g loop
           });  // a loop
        });
        RP_CALI_SUBKERNEL_END("FEMSWEEP_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n FEMSWEEP : Unknown Sequential variant id = " << vid << std::endl;
    }

  }

}

void FEMSWEEP::defineSeqVariantTunings()
{
  for (VariantID vid : {Base_Seq, RAJA_Seq}) {
    addVariantTuning<&FEMSWEEP::runSeqVariant<false>>(
        vid, getDefaultTuningName());

    addVariantTuning<&FEMSWEEP::runSeqVariant<true>>(
        vid, FEMSWEEP_UNROLL_TUNING_NAME);
  }
}

} // end namespace apps
} // end namespace rajaperf
