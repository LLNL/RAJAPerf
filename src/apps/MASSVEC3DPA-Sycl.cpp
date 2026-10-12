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

#include "MASSVEC3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void MASSVEC3DPA::runSyclVariantImpl(VariantID vid)
{

  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASSVEC3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D);
  const ::sycl::range<3> gridSize(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D*NE);

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
      qu.submit([&](::sycl::handler& h) {

        constexpr Index_type MQ1 = mvpa::Q1D;
        constexpr Index_type MD1 = mvpa::D1D;
        constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        auto smB_vec  = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1 * MD1), h);
        auto smBt_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MD1 * MQ1), h);
        auto sm0_vec  = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);
        auto sm1_vec  = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr smB_arr = smB_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr smBt_arr = smBt_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_type(*smB)[MD1] = (Real_type(*)[MD1])smB_arr;
             Real_type(*smBt)[MQ1] = (Real_type(*)[MQ1])smBt_arr;

             Real_type(*smX)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
             Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
             Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
             Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
             Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
             Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

             SYCL_SHARED_LOOP_2D(q, d, mvpa::Q1D, mvpa::D1D) {
               MASSVEC3DPA_1;
             }

             for (int c = 0; c < 3; ++c) {
               SYCL_SHARED_LOOP_3D(dx, dy, dz, mvpa::D1D, mvpa::D1D, mvpa::D1D) {

                 MASSVEC3DPA_2;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(qx, dy, dz, mvpa::Q1D, mvpa::D1D, mvpa::D1D) {

                 MASSVEC3DPA_3;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(qx, qy, dz, mvpa::Q1D, mvpa::Q1D, mvpa::D1D) {
                 MASSVEC3DPA_4;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(qx, qy, qz, mvpa::Q1D, mvpa::Q1D, mvpa::Q1D) {
                 MASSVEC3DPA_5;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(dx, qy, qz, mvpa::D1D, mvpa::Q1D, mvpa::Q1D) {
                 MASSVEC3DPA_6;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(dx, dy, qz, mvpa::D1D, mvpa::D1D, mvpa::Q1D) {
                 MASSVEC3DPA_7;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

               SYCL_SHARED_LOOP_3D(dx, dy, dz, mvpa::D1D, mvpa::D1D, mvpa::D1D) {
                 MASSVEC3DPA_8;
               }
               itm.barrier(::sycl::access::fence_space::local_space);

             }

           });
      });
      RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

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
      constexpr int MQ1 = mvpa::Q1D;
      constexpr int MD1 = mvpa::D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      constexpr int no_mats = 2;
      shmem += MQ1 * MD1 * no_mats * MDQ * MDQ * MDQ * sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASSVEC3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(mvpa::Q1D, mvpa::Q1D, mvpa::Q1D), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr Index_type MQ1 = mvpa::Q1D;
              constexpr Index_type MD1 = mvpa::D1D;
              constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

              Real_ptr smB_arr  = ctx.getSharedMemory<Real_type>(MQ1 * MD1);
              Real_ptr smBt_arr = ctx.getSharedMemory<Real_type>(MQ1 * MD1);
              Real_ptr sm0 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);
              Real_ptr sm1 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);

              Real_type(*smB)[MD1] = (Real_type(*)[MD1])smB_arr;
              Real_type(*smBt)[MQ1] = (Real_type(*)[MQ1])smBt_arr;

              Real_type(*smX)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
              Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
              Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
              Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
              Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
              Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;


            //3 loops to remain consistent with the GPU versions
            //Masking out of the z-dimension thread is done with GPU versions
            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1), [&](int ) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int d) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int q) {
                MASSVEC3DPA_1;
                });
              });
            });

            for (int c = 0; c < 3; ++c) {

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dx) {
                  MASSVEC3DPA_2;
                });
              });
            });

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qx) {
                  MASSVEC3DPA_3;
                });
              });
            });
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qx) {
                MASSVEC3DPA_4;
                });
              });
            });
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qx) {

                MASSVEC3DPA_5;
                });
              });
            });
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dx) {

                MASSVEC3DPA_6;
                });
              });
            });

            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::Q1D), [&](int qz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dx) {

                MASSVEC3DPA_7;
                });
              });
            });
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dz) {
              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dy) {
                RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mvpa::D1D), [&](int dx) {
                  MASSVEC3DPA_8;
                });
              });
            });

            ctx.teamSync();

            } //c - dim loop

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASSVEC3DPA_1");

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASSVEC3DPA : Unknown Sycl variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASSVEC3DPA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
