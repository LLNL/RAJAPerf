//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
void MASS3DPA::runSyclVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DPA_DATA_SETUP;

  const ::sycl::range<3> blockSize(MPA_Q1D, MPA_Q1D, 1);
  const ::sycl::range<3> gridSize(NE*MPA_Q1D,MPA_Q1D,1);

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&](cl::sycl::handler& h) {

        constexpr int MQ1 = MPA_Q1D;
        constexpr int MD1 = MPA_D1D;
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        auto sDQ_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MQ1 * MD1), h);
        auto sm0_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);
        auto sm1_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, blockSize),
           [=] (cl::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(0);

             double *sDQ = sDQ_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             double(*Bsmem)[MD1] = (double(*)[MD1])sDQ;
             double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ;

             double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0;
             double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1;
             double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0;
             double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1;
             double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0;
             double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

             SYCL_FOREACH_THREAD(dy, 1, MPA_D1D) {
               SYCL_FOREACH_THREAD(dx, 0, MPA_D1D){
                 MASS3DPA_1
               }
               SYCL_FOREACH_THREAD(dx, 0, MPA_Q1D) {
                 MASS3DPA_2
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dy, 1, MPA_D1D) {
               SYCL_FOREACH_THREAD(qx, 0, MPA_Q1D) {
                 MASS3DPA_3
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MPA_Q1D) {
               SYCL_FOREACH_THREAD(qx, 0, MPA_Q1D) {
                 MASS3DPA_4
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MPA_Q1D) {
               SYCL_FOREACH_THREAD(qx, 0, MPA_Q1D) {
                 MASS3DPA_5
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(d, 1, MPA_D1D) {
               SYCL_FOREACH_THREAD(q, 0, MPA_Q1D) {
                 MASS3DPA_6
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MPA_Q1D) {
               SYCL_FOREACH_THREAD(dx, 0, MPA_D1D) {
                 MASS3DPA_7
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dy, 1, MPA_D1D) {
               SYCL_FOREACH_THREAD(dx, 0, MPA_D1D) {
                 MASS3DPA_8
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dy, 1, MPA_D1D) {
               SYCL_FOREACH_THREAD(dx, 0, MPA_D1D) {
                 MASS3DPA_9
               }
             }

           });
      });

    }
    stopTimer();

    break;
  }

  case RAJA_SYCL: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x = RAJA::LoopPolicy<RAJA::sycl_group_0_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::sycl_local_0_direct>;

    using inner_y = RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr int MQ1 = MPA_Q1D;
      constexpr int MD1 = MPA_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      constexpr int no_mats = 2;
      shmem += MQ1 * MD1 * no_mats * MDQ * MDQ * MDQ * sizeof(double);
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(MPA_Q1D, MPA_Q1D, 1), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

             //Redefine inside the lambda to keep consistent with base version
             constexpr int MQ1 = MPA_Q1D;
             constexpr int MD1 = MPA_D1D;
             constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

             double *sDQ = ctx.getSharedMemory<double>(MQ1 * MD1);
             double *sm0 = ctx.getSharedMemory<double>(MDQ * MDQ * MDQ);
             double *sm1 = ctx.getSharedMemory<double>(MDQ * MDQ * MDQ);

             double(*Bsmem)[MD1] = (double(*)[MD1])sDQ;
             double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ;

             double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0;
             double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1;
             double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0;
             double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1;
             double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0;
             double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_1
                    }
                  );  // RAJA::loop<inner_x>

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int dx) {
                      MASS3DPA_2
                    }
                  );  // RAJA::loop<inner_x>
                }  // lambda (dy)
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_3
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_4
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_5
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int d) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int q) {
                      MASS3DPA_6
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_7
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_8
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_9
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DPA : Unknown Sycl variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
