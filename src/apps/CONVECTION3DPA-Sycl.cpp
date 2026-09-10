//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void CONVECTION3DPA::runSyclVariantImpl(VariantID vid) {
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  CONVECTION3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(conv::Q1D, conv::Q1D, conv::Q1D);
  const ::sycl::range<3> gridSize(conv::Q1D,conv::Q1D,conv::Q1D*NE);

  constexpr size_t shmem = 0;

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      qu.submit([&](::sycl::handler& h) {

        constexpr Index_type max_D1D = conv::D1D;
        constexpr Index_type max_Q1D = conv::Q1D;
        constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

        auto sm0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm2_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm3_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm4_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm5_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm2 = sm2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm3 = sm3_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm4 = sm4_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm5 = sm5_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_type (*u)[max_D1D][max_D1D] = (Real_type (*)[max_D1D][max_D1D]) sm0;
             Real_type (*Bu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm1;
             Real_type (*Gu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm2;
             Real_type (*BBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
             Real_type (*GBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
             Real_type (*BGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm5;
             Real_type (*GBBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm0;
             Real_type (*BGBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm1;
             Real_type (*BBGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm2;
             Real_type (*DGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
             Real_type (*BDGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
             Real_type (*BBDGu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm5;

             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,conv::D1D)
                 {
                   CONVECTION3DPA_1;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
                 {
                   CONVECTION3DPA_2;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
                 {
                   CONVECTION3DPA_3;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qz,0,conv::Q1D)
                 {
                   CONVECTION3DPA_4;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qz,0,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
                 {
                   CONVECTION3DPA_5;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(dz,0,conv::D1D)
                 {
                   CONVECTION3DPA_6;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(dy,1,conv::D1D)
                 {
                   CONVECTION3DPA_7;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,conv::D1D)
                 {
                   CONVECTION3DPA_8;
                 }
               }
             }
           });

      });
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");


    }
    stopTimer();

    break;
  }

  case RAJA_SYCL: {

    constexpr bool async = true;

    using launch_policy =
      RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x =
      RAJA::LoopPolicy<RAJA::sycl_group_2_loop>;

    using inner_x =
      RAJA::LoopPolicy<RAJA::sycl_local_2_loop>;

    using inner_y =
      RAJA::LoopPolicy<RAJA::sycl_local_1_loop>;

    using inner_z =
      RAJA::LoopPolicy<RAJA::sycl_local_0_loop>;

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr Index_type max_D1D = conv::D1D;
      constexpr Index_type max_Q1D = conv::Q1D;
      constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

      constexpr Index_type no_mats = 6;
      shmem += max_DQ*max_DQ*max_DQ  * no_mats * sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                             RAJA::Threads(conv::Q1D, conv::Q1D, conv::Q1D), shmem),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr Index_type max_D1D = conv::D1D;
              constexpr Index_type max_Q1D = conv::Q1D;
              constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

              Real_ptr sm0 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm1 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm2 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm3 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm4 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm5 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);

              Real_type (*u)[max_D1D][max_D1D] = (Real_type (*)[max_D1D][max_D1D]) sm0;
              Real_type (*Bu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm1;
              Real_type (*Gu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm2;
              Real_type (*BBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
              Real_type (*GBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
              Real_type (*BGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm5;
              Real_type (*GBBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm0;
              Real_type (*BGBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm1;
              Real_type (*BBGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm2;
              Real_type (*DGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
              Real_type (*BDGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
              Real_type (*BBDGu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm5;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_8;

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
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n CONVECTION3DPA : Unknown Sycl variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
