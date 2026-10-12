//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_SYCL: {

    const ::sycl::range<3> workGroupSize(mea::Q1D, mea::Q1D, mea::Q1D);
    const ::sycl::range<3> gridSize(mea::Q1D,mea::Q1D,mea::Q1D*NE);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      constexpr size_t shmem = 0;
      qu.submit([&](::sycl::handler& h) {

      ::sycl::local_accessor<Real_type, 2> s_B(::sycl::range<2>(mea::Q1D,mea::D1D),h);
      ::sycl::local_accessor<Real_type, 3> s_D(::sycl::range<3>(mea::Q1D,mea::Q1D,mea::Q1D),h);

      h.parallel_for
        (::sycl::nd_range<3>(gridSize, workGroupSize),
         [=] (::sycl::nd_item<3> itm) {

           const Index_type e = itm.get_group(2);

           SYCL_FOREACH_THREAD(iz, 0, 1) {
             SYCL_FOREACH_THREAD(d, 2, mea::D1D) {
               SYCL_FOREACH_THREAD(q, 1, mea::Q1D) {
                 MASS3DEA_1
               }
             }
           }

           //not needed as we dynamicaly allocate shared memory in sycl
           //MASS3DEA_2

           SYCL_FOREACH_THREAD(k1, 2, mea::Q1D) {
             SYCL_FOREACH_THREAD(k2, 1, mea::Q1D) {
               SYCL_FOREACH_THREAD(k3, 0, mea::Q1D) {
                 MASS3DEA_3
               }
             }
           }

           itm.barrier(::sycl::access::fence_space::local_space);

           SYCL_FOREACH_THREAD(i1, 2, mea::D1D) {
             SYCL_FOREACH_THREAD(i2, 1, mea::D1D) {
               SYCL_FOREACH_THREAD(i3, 0, mea::D1D) {
                 MASS3DEA_4
               }
             }
           }

         });
      });
      RP_CALI_SUBKERNEL_END("MASS3DEA_1");

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

    constexpr size_t shmem = (mea::Q1D*mea::D1D + mea::Q1D*mea::Q1D*mea::Q1D)*sizeof(Real_type);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(mea::D1D, mea::D1D, mea::D1D), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              Real_ptr s_B_ptr = ctx.getSharedMemory<Real_type>(mea::Q1D*mea::D1D);
              Real_ptr s_D_ptr = ctx.getSharedMemory<Real_type>(mea::Q1D*mea::Q1D*mea::Q1D);

              Real_type (*s_B)[mea::D1D] = (Real_type (*)[mea::D1D]) s_B_ptr;
              Real_type (*s_D)[mea::Q1D][mea::Q1D] = (Real_type (*)[mea::Q1D][mea::Q1D]) s_B_ptr;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>

              //not needed as we dynamicaly allocate shared memory in sycl
              //MASS3DEA_2

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                [&](Index_type k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                    [&](Index_type k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type k3) {
                          MASS3DEA_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                [&](Index_type i1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type i2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::D1D),
                        [&](Index_type i3) {
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
      //clang-format on      
      RP_CALI_SUBKERNEL_END("MASS3DEA_1");

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

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DEA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
