//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void MASS3DEA::runSyclVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_SYCL: {

    const ::sycl::range<3> workGroupSize(MEA_Q1D, MEA_Q1D, MEA_Q1D);
    const ::sycl::range<3> gridSize(MEA_Q1D,MEA_Q1D,MEA_Q1D*NE);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      constexpr size_t shmem = 0;
      qu->submit([&](cl::sycl::handler& h) {

      ::sycl::local_accessor<double, 2> s_B(::sycl::range<2>(MEA_Q1D,MEA_D1D),h);
      ::sycl::local_accessor<double, 3> s_D(::sycl::range<3>(MEA_Q1D,MEA_Q1D,MEA_Q1D),h);

      h.parallel_for
        (cl::sycl::nd_range<3>(gridSize, workGroupSize),
         [=] (cl::sycl::nd_item<3> itm) {

           const Index_type e = itm.get_group(2);

           SYCL_FOREACH_THREAD(iz, 0, 1) {
             SYCL_FOREACH_THREAD(d, 2, MEA_D1D) {
               SYCL_FOREACH_THREAD(q, 1, MEA_Q1D) {
                 MASS3DEA_1
               }
             }
           }

           //not needed as we dynamicaly allocate shared memory in sycl
           //MASS3DEA_2

           SYCL_FOREACH_THREAD(k1, 2, MEA_Q1D) {
             SYCL_FOREACH_THREAD(k2, 1, MEA_Q1D) {
               SYCL_FOREACH_THREAD(k3, 0, MEA_Q1D) {
                 MASS3DEA_3
               }
             }
           }

           itm.barrier(::sycl::access::fence_space::local_space);

           SYCL_FOREACH_THREAD(i1, 2, MEA_D1D) {
             SYCL_FOREACH_THREAD(i2, 1, MEA_D1D) {
               SYCL_FOREACH_THREAD(i3, 0, MEA_D1D) {
                 MASS3DEA_4
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

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x = RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::sycl_local_2_loop>;

    using inner_y = RAJA::LoopPolicy<RAJA::sycl_local_1_loop>;

    using inner_z = RAJA::LoopPolicy<RAJA::sycl_local_0_loop>;

    constexpr size_t shmem = (MEA_Q1D*MEA_D1D + MEA_Q1D*MEA_Q1D*MEA_Q1D)*sizeof(double);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(MEA_D1D, MEA_D1D, MEA_D1D), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              double * s_B_ptr = ctx.getSharedMemory<double>(MEA_Q1D*MEA_D1D);
              double * s_D_ptr = ctx.getSharedMemory<double>(MEA_Q1D*MEA_Q1D*MEA_Q1D);

              double (*s_B)[MEA_D1D] = (double (*)[MEA_D1D]) s_B_ptr;
              double (*s_D)[MEA_Q1D][MEA_Q1D] = (double (*)[MEA_Q1D][MEA_Q1D]) s_B_ptr;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>

              //not needed as we dynamicaly allocate shared memory in sycl
              //MASS3DEA_2

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                [&](int k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                    [&](int k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int k3) {
                          MASS3DEA_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                [&](int i1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int i2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                        [&](int i3) {
                          MASS3DEA_4
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DEA : Unknown Sycl variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DEA, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
