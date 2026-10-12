//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

template < size_t work_group_size >
void DIFFUSION3DPA::runSyclVariantImpl(VariantID vid) {
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_SYCL: {

    const ::sycl::range<3> workGroupSize(diff::Q1D, diff::Q1D, diff::Q1D);
    const ::sycl::range<3> gridSize(diff::Q1D,diff::Q1D,diff::Q1D*NE);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      qu.submit([&](::sycl::handler& h) {

        constexpr Index_type MQ1 = diff::Q1D;
        constexpr Index_type MD1 = diff::D1D;
        constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        auto sBG_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1*MD1), h);

        auto sm0_0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm0_1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm0_2_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);
        auto sm1_2_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ*MDQ*MDQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr sBG = sBG_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_ptr sm0_0 = sm0_0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm0_1 = sm0_1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm0_2 = sm0_2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1_0 = sm1_0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1_1 = sm1_1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1_2 = sm1_2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_type (*B)[MD1] = (Real_type (*)[MD1]) sBG;
             Real_type (*G)[MD1] = (Real_type (*)[MD1]) sBG;
             Real_type (*Bt)[MQ1] = (Real_type (*)[MQ1]) sBG;
             Real_type (*Gt)[MQ1] = (Real_type (*)[MQ1]) sBG;

             Real_type (*s_X)[MD1][MD1]    = (Real_type (*)[MD1][MD1]) (sm0_2);
             Real_type (*DDQ0)[MD1][MQ1]   = (Real_type (*)[MD1][MQ1]) (sm0_0);
             Real_type (*DDQ1)[MD1][MQ1]   = (Real_type (*)[MD1][MQ1]) (sm0_1);
             Real_type (*DQQ0)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_0);
             Real_type (*DQQ1)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_1);
             Real_type (*DQQ2)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_2);
             Real_type (*QQQ0)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_0);
             Real_type (*QQQ1)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_1);
             Real_type (*QQQ2)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_2);
             Real_type (*QQD0)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_0);
             Real_type (*QQD1)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_1);
             Real_type (*QQD2)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_2);
             Real_type (*QDD0)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_0);
             Real_type (*QDD1)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_1);
             Real_type (*QDD2)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_2);

             SYCL_FOREACH_THREAD(dz, 0, diff::D1D) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, diff::D1D) {
                   DIFFUSION3DPA_1;
                 }
               }
             }
             if (itm.get_local_id(0) == 0) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(qx, 2, diff::Q1D) {
                   DIFFUSION3DPA_2;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, diff::D1D) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(qx, 2, diff::Q1D) {
                   DIFFUSION3DPA_3;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, diff::D1D) {
               SYCL_FOREACH_THREAD(qy, 1, diff::Q1D) {
                 SYCL_FOREACH_THREAD(qx, 2, diff::Q1D) {
                   DIFFUSION3DPA_4;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, diff::Q1D) {
               SYCL_FOREACH_THREAD(qy, 1, diff::Q1D) {
                 SYCL_FOREACH_THREAD(qx, 2, diff::Q1D) {
                   DIFFUSION3DPA_5;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             if (itm.get_local_id(0) == 0) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(qx, 2, diff::Q1D) {
                   DIFFUSION3DPA_6;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, diff::Q1D) {
               SYCL_FOREACH_THREAD(qy, 1, diff::Q1D) {
                 SYCL_FOREACH_THREAD(dx, 2, diff::D1D) {
                   DIFFUSION3DPA_7;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qz, 0, diff::Q1D) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, diff::D1D) {
                   DIFFUSION3DPA_8;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dz, 0, diff::D1D) {
               SYCL_FOREACH_THREAD(dy, 1, diff::D1D) {
                 SYCL_FOREACH_THREAD(dx, 2, diff::D1D) {
                   DIFFUSION3DPA_9;
                 }
               }
             }

           });
        });
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");


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
      constexpr Index_type MQ1 = diff::Q1D;
      constexpr Index_type MD1 = diff::D1D;
      constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      const size_t local_mats = 6;
      shmem += MQ1*MD1*sizeof(Real_type) + local_mats*MDQ*MDQ*MDQ*sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
                             RAJA::LaunchParams(RAJA::Teams(NE),
                             RAJA::Threads(diff::Q1D, diff::Q1D, diff::Q1D), shmem),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

            const bool symmetric = true;

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr Index_type MQ1 = diff::Q1D;
              constexpr Index_type MD1 = diff::D1D;
              constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

              Real_ptr sBG = ctx.getSharedMemory<Real_type>(MQ1*MD1);
              Real_ptr sm0_0 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);
              Real_ptr sm0_1 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);
              Real_ptr sm0_2 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);
              Real_ptr sm1_0 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);
              Real_ptr sm1_1 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);
              Real_ptr sm1_2 = ctx.getSharedMemory<Real_type>(MDQ*MDQ*MDQ);

             Real_type (*B)[MD1] = (Real_type (*)[MD1]) sBG;
             Real_type (*G)[MD1] = (Real_type (*)[MD1]) sBG;
             Real_type (*Bt)[MQ1] = (Real_type (*)[MQ1]) sBG;
             Real_type (*Gt)[MQ1] = (Real_type (*)[MQ1]) sBG;

             Real_type (*s_X)[MD1][MD1]    = (Real_type (*)[MD1][MD1]) (sm0_2);
             Real_type (*DDQ0)[MD1][MQ1]   = (Real_type (*)[MD1][MQ1]) (sm0_0);
             Real_type (*DDQ1)[MD1][MQ1]   = (Real_type (*)[MD1][MQ1]) (sm0_1);
             Real_type (*DQQ0)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_0);
             Real_type (*DQQ1)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_1);
             Real_type (*DQQ2)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm1_2);
             Real_type (*QQQ0)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_0);
             Real_type (*QQQ1)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_1);
             Real_type (*QQQ2)[MQ1][MQ1]   = (Real_type (*)[MQ1][MQ1]) (sm0_2);
             Real_type (*QQD0)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_0);
             Real_type (*QQD1)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_1);
             Real_type (*QQD2)[MQ1][MD1]   = (Real_type (*)[MQ1][MD1]) (sm1_2);
             Real_type (*QDD0)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_0);
             Real_type (*QDD1)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_1);
             Real_type (*QDD2)[MD1][MD1]   = (Real_type (*)[MD1][MD1]) (sm0_2);

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                        [&](Index_type dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                          DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
               [&](Index_type dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

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
      //clang-format on
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");
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

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DIFFUSION3DPA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
