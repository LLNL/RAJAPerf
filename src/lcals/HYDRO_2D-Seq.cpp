//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void HYDRO_2D::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY1;
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY2;
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY3;
          }
        }

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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam1(k, j);
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam2(k, j);
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            hydro2d_base_lam3(k, j);
          }
        }

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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam1);

        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam2);

        RAJA::kernel_resource<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     res,
                     hydro2d_lam3);

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

} // end namespace lcals
} // end namespace rajaperf
