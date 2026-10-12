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


void HYDRO_2D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY1;
	    }
	  }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY2;
	    }
	  }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY3;
	    }
	  }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

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

        #pragma omp parallel
        {

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam1(k, j);
            }
          }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam2(k, j);
            }
          }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

          RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam3(k, j);
            }
          }
          RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

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
          RAJA::statement::For<0, RAJA::omp_for_nowait_static_exec< >,  // k
            RAJA::statement::For<1, RAJA::seq_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

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

        }); // end omp parallel region

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace lcals
} // end namespace rajaperf
