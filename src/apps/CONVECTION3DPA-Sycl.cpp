//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  CONVECTION3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(CPA_Q1D, CPA_Q1D, CPA_Q1D);
  const ::sycl::range<3> gridSize(CPA_Q1D,CPA_Q1D,CPA_Q1D*NE);

  constexpr size_t shmem = 0;

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&](cl::sycl::handler& h) {

        constexpr int max_D1D = CPA_D1D;
        constexpr int max_Q1D = CPA_Q1D;
        constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

        auto sm0_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm1_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm2_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm3_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm4_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm5_vec = ::sycl::local_accessor<double, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (cl::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             double *sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm2 = sm2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm3 = sm3_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm4 = sm4_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             double *sm5 = sm5_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
             double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
             double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
             double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
             double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
             double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
             double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
             double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
             double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
             double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
             double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
             double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;

             SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,CPA_D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,CPA_D1D)
                 {
                   CONVECTION3DPA_1;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,CPA_D1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
                 {
                   CONVECTION3DPA_2;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
               {
                 SYCL_FOREACH_THREAD(qy,1,CPA_Q1D)
                 {
                   CONVECTION3DPA_3;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,CPA_Q1D)
               {
                 SYCL_FOREACH_THREAD(qz,0,CPA_Q1D)
                 {
                   CONVECTION3DPA_4;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qz,0,CPA_Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,CPA_Q1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
                 {
                   CONVECTION3DPA_5;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,CPA_Q1D)
               {
                 SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
                 {
                   CONVECTION3DPA_6;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,CPA_Q1D)
               {
                 SYCL_FOREACH_THREAD(dy,1,CPA_D1D)
                 {
                   CONVECTION3DPA_7;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dz,0,CPA_D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,CPA_D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,CPA_D1D)
                 {
                   CONVECTION3DPA_8;
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
      constexpr int max_D1D = CPA_D1D;
      constexpr int max_Q1D = CPA_Q1D;
      constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

      constexpr int no_mats = 6;
      shmem += max_DQ*max_DQ*max_DQ  * no_mats * sizeof(double);
    }

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                             RAJA::Threads(CPA_Q1D, CPA_Q1D, CPA_Q1D), shmem),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr int max_D1D = CPA_D1D;
              constexpr int max_Q1D = CPA_Q1D;
              constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

              double * sm0 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);
              double * sm1 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);
              double * sm2 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);
              double * sm3 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);
              double * sm4 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);
              double * sm5 = ctx.getSharedMemory<double>(max_DQ*max_DQ*max_DQ);

              double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
              double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
              double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
              double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
              double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
              double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
              double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
              double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
              double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
              double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
              double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
              double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

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

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
