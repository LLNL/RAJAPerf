//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define CPU_FOREACH(i, k, N) for (int i = 0; i < N; i++)

void MASS3DPA::runStdParVariant(VariantID vid, size_t tune_idx) {
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_StdPar: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      std::for_each_n( std::execution::par_unseq,
                       counting_iterator<int>(0), NE, 
                       [=](int e) {

        MASS3DPA_0_CPU

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D){
            MASS3DPA_1
          }
          CPU_FOREACH(dx, x, MPA_Q1D) {
            MASS3DPA_2
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_3
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_4
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_5
          }
        }

        CPU_FOREACH(d, y, MPA_D1D) {
          CPU_FOREACH(q, x, MPA_Q1D) {
            MASS3DPA_6
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_7
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_8
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_9
          }
        }

      }); // element loop

    }
    stopTimer();

    break;
  }

  default:
    getCout() << "\n MASS3DPA : Unknown StdPar variant id = " << vid << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

