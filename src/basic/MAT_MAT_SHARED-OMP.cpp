//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_MAT_SHARED::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  MAT_MAT_SHARED_DATA_SETUP;

  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel
      {
#pragma omp for
        for (Index_type by = 0; by < Ny; ++by) {
          for (Index_type bx = 0; bx < Nx; ++bx) {

            //Work around for when compiling with CLANG and HIP
            //See notes in MAT_MAT_SHARED.hpp
            MAT_MAT_SHARED_BODY_0_CLANG_HIP_CPU(TL_SZ)

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
    }
    stopTimer();

    break;
  }

  case Lambda_OpenMP: {

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

#pragma omp parallel for
      for (Index_type by = 0; by < Ny; ++by) {
        outer_y(by);
      }
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::omp_for_exec>;

    using outer_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //Grid is empty as the host does not need a compute grid to be specified
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  MAT_MAT_SHARED_BODY_0(TL_SZ)

                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                    [&](Index_type ty) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, TL_SZ),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_1(TL_SZ)
                        }
                      );  // RAJA::loop<inner_x>
                    }
                  );  // RAJA::loop<inner_y

                  for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

                    RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                      [&](Index_type ty) {
                        RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, TL_SZ),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_2(TL_SZ)
                          }
                        );  // RAJA::loop<inner_x>
                      }
                    );  // RAJA::loop<inner_y>

                    ctx.teamSync();

                    RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                      [&](Index_type ty) {
                        RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, TL_SZ),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_3(TL_SZ)
                          }
                        );  // RAJA::loop<inner_x>
                      }
                    );  // RAJA::loop<inner_y>

                    ctx.teamSync();

                  }  // for (k)

                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                    [&](Index_type ty) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, TL_SZ),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_4(TL_SZ)
                        }
                      );  // RAJA::loop<inner_x>
                    }
                  );  //  RAJA::loop<inner_y>

                }  // lambda (bx)
              );  // RAJA::loop<outer_x>
            }  // lambda (by)
          );  // RAJA::loop<outer_y>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {
    getCout() << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid
              << std::endl;
  }
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace basic
} // end namespace rajaperf
