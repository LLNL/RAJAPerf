//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template < size_t work_group_size >
void MAT_MAT::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  constexpr Index_type tile_size = integer::sqrt(work_group_size);
  static_assert(tile_size*tile_size == work_group_size, "Invalid block_size");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, tile_size);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, tile_size);

  //Right most is the fastest index
  const ::sycl::range<3> workGroupSize(1, tile_size, tile_size);
  const ::sycl::range<3> gridSize(1, Ny*tile_size, Nx*tile_size);

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MAT_MAT_DATA_SETUP;

  if (vid == Base_SYCL) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MAT_MAT_1");
      qu.submit([&](::sycl::handler& h) {

        //No local accessors are needed as nothing is shared between
        //work-items in this variant

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             Index_type tx = itm.get_local_id(2);
             Index_type ty = itm.get_local_id(1);
             Index_type bx = itm.get_group(2);
             Index_type by = itm.get_group(1);

             MAT_MAT_BODY_1(tile_size)

               for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

                 MAT_MAT_BODY_2(tile_size)
               }

             MAT_MAT_BODY_3(tile_size)

           });

      });
      RP_CALI_SUBKERNEL_END("MAT_MAT_1");


    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using teams_x = RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using teams_y = RAJA::LoopPolicy<RAJA::sycl_group_1_direct>;

    using threads_x = RAJA::LoopPolicy<RAJA::sycl_local_2_direct>;

    using threads_y = RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MAT_MAT_1");
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(Nx, Ny),
                           RAJA::Threads(tile_size, tile_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {

                          MAT_MAT_BODY_1(tile_size)

                          for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {
                            MAT_MAT_BODY_2(tile_size)
                          }  // for (k)

                          MAT_MAT_BODY_3(tile_size)
                        }
                      );  // RAJA::loop<threads_x>
                    }
                  );  // RAJA::loop<threads_y>

                }  // lambda (bx)
              );  // RAJA::loop<teams_x>
            }  // lambda (by)
          );  // RAJA::loop<teams_y>

        }   // outer lambda (ctx)
      );  // RAJA::launch
      RP_CALI_SUBKERNEL_END("MAT_MAT_1");

    }  // loop over kernel reps
    stopTimer();

  } else {
    getCout() << "\n  MAT_MAT : Unknown Sycl variant id = " << vid
              << std::endl;
  }

}


RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MAT_MAT, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
