//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#if defined(BUILD_STDPAR)

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

// This is used below, which is bad for GPU
//#define CPU_FOREACH(i, k, N) for (int i = 0; i < N; i++)

namespace rajaperf {
namespace apps {

void DIFFUSION3DPA::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_StdPar: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      std::for_each_n( std::execution::par_unseq,
                       counting_iterator<int>(0), NE,
                       [=](int e) {

        DIFFUSION3DPA_0_CPU;

        CPU_FOREACH(dz, z, DPA_D1D) {
          CPU_FOREACH(dy, y, DPA_D1D) {
            CPU_FOREACH(dx, x, DPA_D1D) {
              DIFFUSION3DPA_1;
            }
          }
        }

        CPU_FOREACH(dy, y, DPA_D1D) {
          CPU_FOREACH(qx, x, DPA_Q1D) {
            DIFFUSION3DPA_2;
          }
        }

        CPU_FOREACH(dz, z, DPA_D1D) {
          CPU_FOREACH(dy, y, DPA_D1D) {
            CPU_FOREACH(qx, x, DPA_Q1D) {
              DIFFUSION3DPA_3;
            }
          }
        }

        CPU_FOREACH(dz, z, DPA_D1D) {
          CPU_FOREACH(qy, y, DPA_Q1D) {
            CPU_FOREACH(qx, x, DPA_Q1D) {
              DIFFUSION3DPA_4;
            }
          }
        }

        CPU_FOREACH(qz, z, DPA_Q1D) {
          CPU_FOREACH(qy, y, DPA_Q1D) {
            CPU_FOREACH(qx, x, DPA_Q1D) {
              DIFFUSION3DPA_5;
            }
          }
        }

        CPU_FOREACH(d, y, DPA_D1D) {
          CPU_FOREACH(q, x, DPA_Q1D) {
            DIFFUSION3DPA_6;
          }
        }

        CPU_FOREACH(qz, z, DPA_Q1D) {
          CPU_FOREACH(qy, y, DPA_Q1D) {
            CPU_FOREACH(dx, x, DPA_D1D) {
              DIFFUSION3DPA_7;
            }
          }
        }

        CPU_FOREACH(qz, z, DPA_Q1D) {
          CPU_FOREACH(dy, y, DPA_D1D) {
            CPU_FOREACH(dx, x, DPA_D1D) {
              DIFFUSION3DPA_8;
            }
          }
        }

        CPU_FOREACH(dz, z, DPA_D1D) {
          CPU_FOREACH(dy, y, DPA_D1D) {
            CPU_FOREACH(dx, x, DPA_D1D) {
              DIFFUSION3DPA_9;
            }
          }
        }

      }); // element loop

    }
    stopTimer();

    break;
  }

  default:
    getCout() << "\n DIFFUSION3DPA : Unknown StdPar variant id = " << vid << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf

#endif  // BUILD_STDPAR

