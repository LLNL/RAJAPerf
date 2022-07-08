//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_MAT_SHARED::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  MAT_MAT_SHARED_DATA_SETUP;
  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);

  switch (vid) {

  case Base_StdPar: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#warning need parallel for
      for (Index_type by = 0; by < Ny; ++by) {
        for (Index_type bx = 0; bx < Nx; ++bx) {

          MAT_MAT_SHARED_BODY_0(TL_SZ)

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_1(TL_SZ)
            }
          }

          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_2(TL_SZ)
              }
            }

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_3(TL_SZ)
              }
            }

          }

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_4(TL_SZ)
            }
          }
        }
      }
    }
    stopTimer();

    break;
  }

  case Lambda_StdPar: {


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto outer_y = [&](Index_type by) {
        auto outer_x = [&](Index_type bx) {

          MAT_MAT_SHARED_BODY_0(TL_SZ)

          auto inner_y_1 = [&](Index_type ty) {
            auto inner_x_1 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_1(TL_SZ) };

            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              if (tx < TL_SZ)
                inner_x_1(tx);
            }
          };

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            if (ty < TL_SZ)
              inner_y_1(ty);
          }

          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            auto inner_y_2 = [&](Index_type ty) {
              auto inner_x_2 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_2(TL_SZ) };

              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                inner_x_2(tx);
              }
            };

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              inner_y_2(ty);
            }

            auto inner_y_3 = [&](Index_type ty) {
              auto inner_x_3 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_3(TL_SZ) };

              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                inner_x_3(tx);
              }
            };

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              inner_y_3(ty);
            }
          }

          auto inner_y_4 = [&](Index_type ty) {
            auto inner_x_4 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_4(TL_SZ) };

            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              inner_x_4(tx);
            }
          };

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            inner_y_4(ty);
          }
        }; // outer_x

        for (Index_type bx = 0; bx < Nx; ++bx) {
          outer_x(bx);
        }
      };

#warning need parallel for
      for (Index_type by = 0; by < Ny; ++by) {
        outer_y(by);
      }
    }
    stopTimer();

    break;
  }

  default: {
    getCout() << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid
              << std::endl;
  }
  }
#endif
}

} // end namespace basic
} // end namespace rajaperf
