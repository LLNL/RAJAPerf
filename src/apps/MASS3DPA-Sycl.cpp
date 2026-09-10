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

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template <Index_type D1D, Index_type Q1D, Index_type TBATCH>
void MASS3DPA::runSyclVariantImpl(VariantID vid) {
  static_assert(TBATCH == 1, "MASS3DPA SYCL does not use z-batching");

  constexpr Index_type MD1 = D1D;
  constexpr Index_type MQ1 = Q1D;
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;
  constexpr size_t work_group_size = MQ1 * MQ1 * TBATCH;
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(1, MQ1, MQ1);
  const ::sycl::range<3> gridSize(1, MQ1, MQ1*NE);

  switch (vid) {

  case Base_SYCL: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_1");
      qu.submit([&](::sycl::handler& h) {

        auto sDQ_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1 * MD1), h);
        auto sm0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);
        auto sm1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr sDQ = sDQ_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_type(*Bsmem)[MD1] = (Real_type(*)[MD1])sDQ;
             Real_type(*Btsmem)[MQ1] = (Real_type(*)[MQ1])sDQ;

             Real_type(*Xsmem)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
             Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
             Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
             Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
             Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
             Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

             SYCL_FOREACH_THREAD(dy, 1, MD1) {
               SYCL_FOREACH_THREAD(dx, 2, MD1){
                 MASS3DPA_1
               }
               SYCL_FOREACH_THREAD(dx, 2, MQ1) {
                 MASS3DPA_2
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dy, 1, MD1) {
               SYCL_FOREACH_THREAD(qx, 2, MQ1) {
                 MASS3DPA_3
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MQ1) {
               SYCL_FOREACH_THREAD(qx, 2, MQ1) {
                 MASS3DPA_4
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MQ1) {
               SYCL_FOREACH_THREAD(qx, 2, MQ1) {
                 MASS3DPA_5
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(d, 1, MD1) {
               SYCL_FOREACH_THREAD(q, 2, MQ1) {
                 MASS3DPA_6
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(qy, 1, MQ1) {
               SYCL_FOREACH_THREAD(dx, 2, MD1) {
                 MASS3DPA_7
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             SYCL_FOREACH_THREAD(dy, 1, MD1) {
               SYCL_FOREACH_THREAD(dx, 2, MD1) {
                 MASS3DPA_8
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);
             SYCL_FOREACH_THREAD(dy, 1, MD1) {
               SYCL_FOREACH_THREAD(dx, 2, MD1) {
                 MASS3DPA_9
               }
             }

           });
      });
      RP_CALI_SUBKERNEL_END("MASS3DPA_1");

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

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr Index_type no_mats = 2;
      shmem += MQ1 * MD1 * no_mats * MDQ * MDQ * MDQ * sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(MQ1, MQ1), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

             Real_ptr sDQ = ctx.getSharedMemory<Real_type>(MQ1 * MD1);
             Real_ptr sm0 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);
             Real_ptr sm1 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);

             Real_type(*Bsmem)[MD1] = (Real_type(*)[MD1])sDQ;
             Real_type(*Btsmem)[MQ1] = (Real_type(*)[MQ1])sDQ;

             Real_type(*Xsmem)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
             Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
             Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
             Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
             Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
             Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MD1),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MD1),
                    [&](Index_type dx) {
                      MASS3DPA_1
                    }
                  );  // RAJA::loop<inner_x>

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MQ1),
                    [&](Index_type dx) {
                      MASS3DPA_2
                    }
                  );  // RAJA::loop<inner_x>
                }  // lambda (dy)
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MD1),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MQ1),
                    [&](Index_type qx) {
                      MASS3DPA_3
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MQ1),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MQ1),
                    [&](Index_type qx) {
                      MASS3DPA_4
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MQ1),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MQ1),
                    [&](Index_type qx) {
                      MASS3DPA_5
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MD1),
                [&](Index_type d) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MQ1),
                    [&](Index_type q) {
                      MASS3DPA_6
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MQ1),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MD1),
                    [&](Index_type dx) {
                      MASS3DPA_7
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MD1),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MD1),
                    [&](Index_type dx) {
                      MASS3DPA_8
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MD1),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MD1),
                    [&](Index_type dx) {
                      MASS3DPA_9
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DPA_1");

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

void MASS3DPA::defineSyclVariantTunings()
{
  for (VariantID vid : {Base_SYCL, RAJA_SYCL}) {
    seq_for(sycl_gpu_block_sizes_type{}, [&](auto block_size) {
      constexpr Index_type TBATCH = 1;
      constexpr size_t block_size_value = decltype(block_size)::value;

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size_value)) {
        addVariantTuning<&MASS3DPA::runSyclVariantImpl<
            mpa::D1D, mpa::Q1D, TBATCH>>(
            vid, "block_" + std::to_string(block_size_value));
      }
    });
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
