//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void HYDRO_2D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;
#if !defined(RUN_RAJA_SEQ)
  RAJA_UNUSED_VAR(kn);   // prevents a compiler warning in the OpenMP target offload build case
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY1;
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY2;
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY3;
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto hydro2d_base_lam1 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY1;
                               };
      auto hydro2d_base_lam2 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY2;
                               };
      auto hydro2d_base_lam3 = [=] (Index_type k, Index_type j) {
                                 HYDRO_2D_BODY3;
                               };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam1(k, j);
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam2(k, j);
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam3(k, j);
          }
        }
        RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      HYDRO_2D_VIEWS_RAJA;

      auto hydro2d_lam1 = [=] (Index_type k, Index_type j) {
                            HYDRO_2D_BODY1_RAJA;
                          };
      auto hydro2d_lam2 = [=] (Index_type k, Index_type j) {
                            HYDRO_2D_BODY2_RAJA;
                          };
      auto hydro2d_lam3 = [=] (Index_type k, Index_type j) {
                            HYDRO_2D_BODY3_RAJA;
                          };

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::seq_exec,    // k
            RAJA::statement::For<1, RAJA::seq_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam1);
        RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam2);
        RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam3);
        RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace lcals
} // end namespace rajaperf
