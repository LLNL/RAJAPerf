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


void HYDRO_2D::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY1;
	    }
	  }

          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY2;
	    }
	  }

          #pragma omp for schedule(static) nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY3;
	    }
	  }

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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam1(k, j);
            }
          }

          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam2(k, j);
            }
          }

          #pragma omp for schedule(static) nowait
          for (Index_type k = kbeg; k < kend; ++k ) {
            for (Index_type j = jbeg; j < jend; ++j ) {
              hydro2d_base_lam3(k, j);
            }
          }

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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

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

} // end namespace lcals
} // end namespace rajaperf
