//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

void HYDRO_2D::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

#ifdef USE_STDPAR_COLLAPSE
  // this is going to run from [(0,0),..]
  // we will add (1,1) later
  const auto nk = kend-kbeg;
  const auto nj = jend-jbeg;
#endif

  HYDRO_2D_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k  = kbeg + kj / nj;
            const auto j  = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            //std::cerr << "JEFF: " << k << "," << j << "\n";
            HYDRO_2D_BODY1;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k = kbeg + kj / nj;
            const auto j = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            HYDRO_2D_BODY2;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k = kbeg + kj / nj;
            const auto j = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            HYDRO_2D_BODY3;
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

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

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k = kbeg + kj / nj;
            const auto j = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            hydro2d_base_lam1(k, j);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k = kbeg + kj / nj;
            const auto j = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            hydro2d_base_lam2(k, j);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk*nj,
                         [=](Index_type kj) {
            const auto k = kbeg + kj / nj;
            const auto j = jbeg + kj % nj;
#else
        std::for_each_n( std::execution::par,
                         counting_iterator<Index_type>(kbeg), kend-kbeg,
                         [=](Index_type k) {
          std::for_each_n( std::execution::unseq,
                           counting_iterator<Index_type>(jbeg), jend-jbeg,
                           [=](Index_type j) {
#endif
            hydro2d_base_lam3(k, j);
#ifndef USE_STDPAR_COLLAPSE
          });
#endif
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // BUILD_STDPAR

