//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
void DIFFUSION3DPA::runSyclVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_SYCL: {

    const ::sycl::range<3> blockSize(DPA_Q1D, DPA_Q1D, DPA_Q1D);
    const ::sycl::range<3> gridSize(DPA_Q1D,DPA_Q1D,DPA_Q1D*NE);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&](cl::sycl::handler& h) {

        constexpr int MQ1 = DPA_Q1D;
        constexpr int MD1 = DPA_D1D;
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        auto sBG_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MQ1*MD1), h);

        auto sm0_0_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm0_1_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm0_2_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_0_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_1_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_2_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);

        sycl::stream out(1024, 256, h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, blockSize),
           [=] (cl::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             double *sBG = sBG_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             double *sm0_0 = sm0_0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm0_1 = sm0_1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm0_2 = sm0_2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm1_0 = sm1_0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm1_1 = sm1_1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm1_2 = sm1_2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             double (*B)[MD1] = (double (*)[MD1]) sBG;
             double (*G)[MD1] = (double (*)[MD1]) sBG;
             double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
             double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;

             double (*s_X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0_2);
             double (*DDQ0)[MD1][MQ1]   = (double (*)[MD1][MQ1]) (sm0_0);
             double (*DDQ1)[MD1][MQ1]   = (double (*)[MD1][MQ1]) (sm0_1);
             double (*DQQ0)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_0);
             double (*DQQ1)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_1);
             double (*DQQ2)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_2);
             double (*QQQ0)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_0);
             double (*QQQ1)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_1);
             double (*QQQ2)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_2);
             double (*QQD0)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_0);
             double (*QQD1)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_1);
             double (*QQD2)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_2);
             double (*QDD0)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_0);
             double (*QDD1)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_1);
             double (*QDD2)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_2);

             SYCL_FOREACH_THREAD(dz, 0, DPA_D1D) {
               SYCL_FOREACH_THREAD(dy, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, DPA_D1D) {
                   DIFFUSION3DPA_1;
                 }
               }
             }

             if (itm.get_local_id(0) == 0)
             {
               SYCL_FOREACH_THREAD(dy, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(qx, 2, DPA_Q1D) {
                   DIFFUSION3DPA_2;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, DPA_D1D) {
               SYCL_FOREACH_THREAD(dy, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(qx, 2, DPA_Q1D) {
                   DIFFUSION3DPA_3;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, DPA_D1D) {
               SYCL_FOREACH_THREAD(qy, 1, DPA_Q1D) {
                 SYCL_FOREACH_THREAD(qx, 2, DPA_Q1D) {
                   DIFFUSION3DPA_4;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, DPA_Q1D) {
               SYCL_FOREACH_THREAD(qy, 1, DPA_Q1D) {
                 SYCL_FOREACH_THREAD(qx, 2, DPA_Q1D) {
                   DIFFUSION3DPA_5;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             if (itm.get_local_id(0) == 0)
               {
               SYCL_FOREACH_THREAD(d, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(q, 2, DPA_Q1D) {
                   DIFFUSION3DPA_6;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, DPA_Q1D) {
               SYCL_FOREACH_THREAD(qy, 1, DPA_Q1D) {
                 SYCL_FOREACH_THREAD(dx, 2, DPA_D1D) {
                   DIFFUSION3DPA_7;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, DPA_Q1D) {
               SYCL_FOREACH_THREAD(dy, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, DPA_D1D) {
                   DIFFUSION3DPA_8;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, DPA_D1D) {
               SYCL_FOREACH_THREAD(dy, 1, DPA_D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, DPA_D1D) {
                   DIFFUSION3DPA_9;
                 }
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

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::sycl_local_2_loop>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::sycl_local_1_loop>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::sycl_local_0_loop>;

    size_t shmem = 0;
    {
      constexpr int MQ1 = DPA_Q1D;
      constexpr int MD1 = DPA_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      const size_t local_mats = 6;
      shmem += MQ1*MD1*sizeof(double) + local_mats*MDQ*MDQ*MDQ*sizeof(double);
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
                             RAJA::LaunchParams(RAJA::Teams(NE),
                             RAJA::Threads(DPA_Q1D, DPA_Q1D, DPA_Q1D), shmem),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

            const bool symmetric = true;

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr int MQ1 = DPA_Q1D;
              constexpr int MD1 = DPA_D1D;
              constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

              double *sBG = ctx.getSharedMemory<double>(MQ1*MD1);
              double *sm0_0 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);
              double *sm0_1 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);
              double *sm0_2 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);
              double *sm1_0 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);
              double *sm1_1 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);
              double *sm1_2 = ctx.getSharedMemory<double>(MDQ*MDQ*MDQ);

             double (*B)[MD1] = (double (*)[MD1]) sBG;
             double (*G)[MD1] = (double (*)[MD1]) sBG;
             double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
             double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;

             double (*s_X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0_2);
             double (*DDQ0)[MD1][MQ1]   = (double (*)[MD1][MQ1]) (sm0_0);
             double (*DDQ1)[MD1][MQ1]   = (double (*)[MD1][MQ1]) (sm0_1);
             double (*DQQ0)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_0);
             double (*DQQ1)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_1);
             double (*DQQ2)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm1_2);
             double (*QQQ0)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_0);
             double (*QQQ1)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_1);
             double (*QQQ2)[MQ1][MQ1]   = (double (*)[MQ1][MQ1]) (sm0_2);
             double (*QQD0)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_0);
             double (*QQD1)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_1);
             double (*QQD2)[MQ1][MD1]   = (double (*)[MQ1][MD1]) (sm1_2);
             double (*QDD0)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_0);
             double (*QDD1)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_1);
             double (*QDD2)[MD1][MD1]   = (double (*)[MD1][MD1]) (sm0_2);

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                        [&](int dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>


              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int qx) {

                         DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](int RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int d) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int q) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
               [&](int dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

            } // lambda (e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n DIFFUSION3DPA : Unknown Sycl variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DIFFUSION3DPA, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
