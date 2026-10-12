//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DPA_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void MASS3DPA_ATOMIC::runSyclVariantImpl(VariantID vid) {
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DPA_ATOMIC_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D);
  const ::sycl::range<3> gridSize(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D*NE);

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      qu.submit([&](::sycl::handler& h) {

        constexpr Index_type MQ1 = mpa_at::Q1D;
        constexpr Index_type MD1 = mpa_at::D1D;
        constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        auto smB_vec  = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1*MD1), h);
        auto smBt_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1*MD1), h);

        auto sm0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);
        auto sm1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        auto thread_dofs_vec = ::sycl::local_accessor<Index_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             auto smB_ptr = smB_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             auto smBt_ptr = smBt_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

              Real_type (*sm_B)[MD1] = (Real_type (*)[MD1]) smB_ptr;
              Real_type (*sm_Bt)[MQ1] = (Real_type (*)[MQ1]) smBt_ptr;

             auto sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             auto sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

              Real_type (*sm_X)[MD1][MD1] = (Real_type (*)[MD1][MD1])sm0;
              Real_type (*DDQ)[MD1][MQ1] = (Real_type (*)[MD1][MQ1])sm1;
              Real_type (*DQQ)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1])sm0;
              Real_type (*QQQ)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1])sm1;
              Real_type (*QQD)[MQ1][MD1] = (Real_type (*)[MQ1][MD1])sm0;
              Real_type (*QDD)[MD1][MD1] = (Real_type (*)[MD1][MD1])sm1;

             auto thread_dofs = thread_dofs_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             SYCL_FOREACH_THREAD_DIRECT(dz, 0, mpa_at::D1D) {
               SYCL_FOREACH_THREAD_DIRECT(dy, 1, mpa_at::D1D) {
                 SYCL_FOREACH_THREAD_DIRECT(dx, 2, mpa_at::D1D) {
                   MASS3DPA_ATOMIC_1;
                 }
               }
             }

             SYCL_FOREACH_THREAD_DIRECT(dz, 0, 1) {
               SYCL_FOREACH_THREAD_DIRECT(d, 1, mpa_at::D1D) {
                 SYCL_FOREACH_THREAD_DIRECT(q, 2, mpa_at::Q1D) {
                   MASS3DPA_ATOMIC_2;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(dz, 0, mpa_at::D1D) {
               SYCL_FOREACH_THREAD_DIRECT(dy, 1, mpa_at::D1D) {
                 SYCL_FOREACH_THREAD_DIRECT(qx, 2, mpa_at::Q1D) {
                   MASS3DPA_ATOMIC_3;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(dz, 0, mpa_at::D1D) {
               SYCL_FOREACH_THREAD_DIRECT(qy, 1, mpa_at::Q1D) {
                 SYCL_FOREACH_THREAD_DIRECT(qx, 2, mpa_at::Q1D) {
                   MASS3DPA_ATOMIC_4;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(qz, 0, mpa_at::Q1D) {
               SYCL_FOREACH_THREAD_DIRECT(qy, 1, mpa_at::Q1D) {
                 SYCL_FOREACH_THREAD_DIRECT(qx, 2, mpa_at::Q1D) {
                   MASS3DPA_ATOMIC_5;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(qz, 0, mpa_at::Q1D) {
               SYCL_FOREACH_THREAD_DIRECT(qy, 1, mpa_at::Q1D) {
                 SYCL_FOREACH_THREAD_DIRECT(dx, 2, mpa_at::D1D) {
                   MASS3DPA_ATOMIC_6;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(qz, 0, mpa_at::Q1D) {
               SYCL_FOREACH_THREAD_DIRECT(dy, 1, mpa_at::D1D) {
                 SYCL_FOREACH_THREAD_DIRECT(dx, 2, mpa_at::D1D) {
                   MASS3DPA_ATOMIC_7;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD_DIRECT(dz, 0, mpa_at::D1D) {
               SYCL_FOREACH_THREAD_DIRECT(dy, 2, mpa_at::D1D) {
                 SYCL_FOREACH_THREAD_DIRECT(dx, 1, mpa_at::D1D) {
                  MASS3DPA_ATOMIC_8;
                  MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_SYCL);
                 }
               }
             }


           });
      });
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }
    stopTimer();

    break;
  }

  case RAJA_SYCL: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x = RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::sycl_local_2_direct>;

    using inner_y = RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

    using inner_z = RAJA::LoopPolicy<RAJA::sycl_local_0_direct>;

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr Index_type MQ1 = mpa_at::Q1D;
      constexpr Index_type MD1 = mpa_at::D1D;
      constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      constexpr Index_type no_mats = 2;
      shmem +=  MQ1 * MD1 * no_mats * sizeof(Real_type) + //B,Bt
        MDQ * MDQ * MDQ * no_mats * sizeof(Real_type) + //sm0,sm1
        MD1 * MD1 * MD1 * sizeof(Index_type); //thread_dofs
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              constexpr int MQ1 = mpa_at::Q1D;
              constexpr int MD1 = mpa_at::D1D;
              constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

              Real_ptr smB_ptr  = ctx.getSharedMemory<Real_type>(MQ1*MD1);
              Real_ptr smBt_ptr = ctx.getSharedMemory<Real_type>(MQ1*MD1);

              Real_type (*sm_B)[MD1] = (Real_type (*)[MD1]) smB_ptr;
              Real_type (*sm_Bt)[MQ1] = (Real_type (*)[MQ1]) smBt_ptr;

              Real_ptr sm0 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);
              Real_ptr sm1 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);

              Real_type (*sm_X)[MD1][MD1] = (Real_type (*)[MD1][MD1])sm0;
              Real_type (*DDQ)[MD1][MQ1] = (Real_type (*)[MD1][MQ1])sm1;
              Real_type (*DQQ)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1])sm0;
              Real_type (*QQQ)[MQ1][MQ1] = (Real_type (*)[MQ1][MQ1])sm1;
              Real_type (*QQD)[MQ1][MD1] = (Real_type (*)[MQ1][MD1])sm0;
              Real_type (*QDD)[MD1][MD1] = (Real_type (*)[MD1][MD1])sm1;

              Index_ptr thread_dofs = ctx.getSharedMemory<Index_type>(MD1 * MD1 * MD1);

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                        MASS3DPA_ATOMIC_1;
                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 ); // RAJA::loop<inner_y>
               } // lambda (dz)
             ); // RAJA::loop<inner_z>

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type ) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                   [&](Index_type d) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                       [&](Index_type q) {
                        MASS3DPA_ATOMIC_2;
                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 ); // RAJA::loop<inner_y>
               } // lambda ()
             ); // RAJA::loop<inner_z>
             ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_6;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_7;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_8;
                      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_RAJA_SYCL);
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DPA_ATOMIC : Unknown Sycl variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA_ATOMIC, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
