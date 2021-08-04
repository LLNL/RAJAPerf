//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#include <ranges>
#include <algorithm>
#include <execution>

#include <iostream>

namespace rajaperf
{
namespace apps
{


void LTIMES::runStdParVariant(VariantID vid)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      auto range = std::views::iota((Index_type)0,num_z);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type z) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                LTIMES_BODY;
              }
            }
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

      auto range = std::views::iota((Index_type)0,num_z);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        std::begin(range), std::end(range),
                        [=](Index_type z) {
          for (Index_type g = 0; g < num_g; ++g ) {
            for (Index_type m = 0; m < num_m; ++m ) {
              for (Index_type d = 0; d < num_d; ++d ) {
                ltimes_base_lam(d, z, g, m);
              }
            }
          }
        });

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_STDPAR)
    case RAJA_StdPar : {

      LTIMES_VIEWS_RANGES_RAJA;

      auto ltimes_lam = [=](ID d, IZ z, IG g, IM m) {
                          LTIMES_BODY_RAJA;
                        };

      using EXEC_POL = 
        RAJA::KernelPolicy<
          RAJA::statement::For<1, RAJA::loop_exec,       // z
            RAJA::statement::For<2, RAJA::loop_exec,     // g
              RAJA::statement::For<3, RAJA::loop_exec,   // m
                RAJA::statement::For<0, RAJA::loop_exec, // d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                                 IZRange(0, num_z),
                                                 IGRange(0, num_g),
                                                 IMRange(0, num_m)), 
                                ltimes_lam
                              );

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_STDPAR

    default : {
      std::cout << "\n LTIMES : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

} // end namespace apps
} // end namespace rajaperf
