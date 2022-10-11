//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void LTIMES::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

#ifdef USE_STDPAR_COLLAPSE
  auto begin = counting_iterator<Index_type>(0);
  auto end   = counting_iterator<Index_type>(num_z*num_g*num_m);
#else
  auto begin = counting_iterator<Index_type>(0);
  auto end   = counting_iterator<Index_type>(num_z);
#endif

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type zgm) {
              const auto z  = zgm / (num_g*num_m);
              const auto gm = zgm % (num_g*num_m);
              const auto g  = gm / num_m;
              const auto m  = gm % num_m;
#else
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type z) {
          for (Index_type g = 0; g < num_g; ++g )
            for (Index_type m = 0; m < num_m; ++m )
#endif
              for (Index_type d = 0; d < num_d; ++d ) {
                LTIMES_BODY;
              }
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      auto ltimes_base_lam = [=](Index_type d, Index_type z, 
                                 Index_type g, Index_type m) {
                               LTIMES_BODY;
                             };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#ifdef USE_STDPAR_COLLAPSE
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type zgm) {
              const auto z  = zgm / (num_g*num_m);
              const auto gm = zgm % (num_g*num_m);
              const auto g  = gm / num_m;
              const auto m  = gm % num_m;
#else
        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type z) {
          for (Index_type g = 0; g < num_g; ++g )
            for (Index_type m = 0; m < num_m; ++m )
#endif
              for (Index_type d = 0; d < num_d; ++d ) {
                ltimes_base_lam(d, z, g, m);
              }
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n LTIMES : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

} // end namespace apps
} // end namespace rajaperf