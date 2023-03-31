//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_STDPAR)

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void CONVECTION3DPA::runStdParVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_StdPar: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      std::for_each_n( std::execution::par_unseq,
                       counting_iterator<int>(0), NE,
                       [=](int e) {

        CONVECTION3DPA_0_CPU;

        CPU_FOREACH(dz,z,CPA_D1D) {
          CPU_FOREACH(dy,y,CPA_D1D) {
            CPU_FOREACH(dx,x,CPA_D1D) {
              CONVECTION3DPA_1;
            }
          }
        }

        CPU_FOREACH(dz,z,CPA_D1D) {
          CPU_FOREACH(dy,y,CPA_D1D) {
            CPU_FOREACH(qx,x,CPA_Q1D) {
              CONVECTION3DPA_2;
            }
          }
        }

        CPU_FOREACH(dz,z,CPA_D1D) {
          CPU_FOREACH(qx,x,CPA_Q1D) {
            CPU_FOREACH(qy,y,CPA_Q1D) {
              CONVECTION3DPA_3;
            }
          }
        }

        CPU_FOREACH(qx,x,CPA_Q1D) {
          CPU_FOREACH(qy,y,CPA_Q1D) {
            CPU_FOREACH(qz,z,CPA_Q1D) {
              CONVECTION3DPA_4;
            }
          }
        }

        CPU_FOREACH(qz,z,CPA_Q1D) {
          CPU_FOREACH(qy,y,CPA_Q1D) {
            CPU_FOREACH(qx,x,CPA_Q1D) {
              CONVECTION3DPA_5;
            }
          }
        }

        CPU_FOREACH(qx,x,CPA_Q1D) {
          CPU_FOREACH(qy,y,CPA_Q1D) {
            CPU_FOREACH(dz,z,CPA_D1D) {
              CONVECTION3DPA_6;
            }
          }
        }

        CPU_FOREACH(dz,z,CPA_D1D) {
           CPU_FOREACH(qx,x,CPA_Q1D) {
              CPU_FOREACH(dy,y,CPA_D1D) {
                CONVECTION3DPA_7;
             }
          }
        }

        CPU_FOREACH(dz,z,CPA_D1D) {
          CPU_FOREACH(dy,y,CPA_D1D) {
            CPU_FOREACH(dx,x,CPA_D1D) {
              CONVECTION3DPA_8;
            }
          }
        }

      }); // element loop

    }
    stopTimer();

    break;
  }

  default:
    getCout() << "\n CONVECTION3DPA : Unknown StdPar variant id = " << vid << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_STDPAR

