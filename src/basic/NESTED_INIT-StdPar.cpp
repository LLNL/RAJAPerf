//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(BUILD_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


void NESTED_INIT::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nj*nk,
                         [=](Index_type ijk) {
              const auto k  = ijk / (nj*ni);
              const auto ij = ijk % (nj*ni);
              const auto j  = ij / ni;
              const auto i  = ij % ni;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk,
                         [=](Index_type k) {
            for (Index_type j = 0; j < nj; ++j )
              for (Index_type i = 0; i < ni; ++i )
#endif
              {
                NESTED_INIT_BODY;
                //getCout() << i << "," << j << "," << k << ";" << ijk << " PAR\n";
              }
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), ni*nj*nk,
                         [=](Index_type ijk) {
              const auto k  = ijk / (nj*ni);
              const auto ij = ijk % (nj*ni);
              const auto j  = ij / ni;
              const auto i  = ij % ni;
#else
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), nk,
                         [=](Index_type k) {
            for (Index_type j = 0; j < nj; ++j )
              for (Index_type i = 0; i < ni; ++i )
#endif
              {
                nestedinit_lam(i, j, k);
              }
          });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf

#endif  // BUILD_STDPAR

