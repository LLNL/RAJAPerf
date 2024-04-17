//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template < size_t block_size >
void MAT_MAT_SHARED::runSyclVariantImpl(VariantID vid)
{
  constexpr Index_type tile_size = integer::sqrt(block_size);
  static_assert(tile_size*tile_size == block_size, "Invalid block_size");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, tile_size);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, tile_size);

  //Right most is the fastest index
  const ::sycl::range<3> blockSize(1, tile_size, tile_size);
  const ::sycl::range<3> gridSize(1, Ny*tile_size, Nx*tile_size);

  constexpr size_t shmem = tile_size * tile_size;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MAT_MAT_SHARED_DATA_SETUP;

  if (vid == Base_SYCL) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&](cl::sycl::handler& h) {

       ::sycl::local_accessor<double, 2> As(::sycl::range<2>(tile_size, tile_size), h);
       ::sycl::local_accessor<double, 2> Bs(::sycl::range<2>(tile_size, tile_size), h);
       ::sycl::local_accessor<double, 2> Cs(::sycl::range<2>(tile_size, tile_size), h);

        h.parallel_for
          (cl::sycl::nd_range<3>(gridSize, blockSize),
           [=] (cl::sycl::nd_item<3> itm) {

             Index_type tx = itm.get_local_id(2);
             Index_type ty = itm.get_local_id(1);
             Index_type bx = itm.get_group(2);
             Index_type by = itm.get_group(1);

             MAT_MAT_SHARED_BODY_1(tile_size)

               for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

                 MAT_MAT_SHARED_BODY_2(tile_size)

                 itm.barrier(::sycl::access::fence_space::local_space);

                 MAT_MAT_SHARED_BODY_3(tile_size)

                 itm.barrier(::sycl::access::fence_space::local_space);
               }

             MAT_MAT_SHARED_BODY_4(tile_size)

           });

      });


    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    constexpr bool async = true;

    const int local_mats = 3;
    constexpr size_t shmem = tile_size * tile_size * local_mats * sizeof(double);

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using teams_x = RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using teams_y = RAJA::LoopPolicy<RAJA::sycl_group_1_direct>;

    using threads_x = RAJA::LoopPolicy<RAJA::sycl_local_2_direct>;

    using threads_y = RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(Nx, Ny),
                           RAJA::Threads(tile_size, tile_size), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  //We only support dynamic shared memory in Sycl
                  //Thus requiring a different setup than other backends
                  //which use static shared memory
                  MAT_MAT_SHARED_BODY_SYCL_0(tile_size)

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_1(tile_size)                  
                        }
                      );  // RAJA::loop<threads_x>
                    }
                  );  // RAJA::loop<threads_y>

                  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

                    RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::loop<threads_x>(ctx,
                                                    RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_2(tile_size)
                          }
                        ); // RAJA::loop<threads_x>
                      }
                    );  // RAJA::loop<threads_y>

                    ctx.teamSync();

                    RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_3(tile_size)
                          }
                        );  // RAJA::loop<threads_x>
                      }
                    );  // RAJA::loop<threads_y>

                    ctx.teamSync();

                  }  // for (k)

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_4(tile_size)
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

    }  // loop over kernel reps
    stopTimer();

  } else {
    getCout() << "\n  MAT_MAT_SHARED : Unknown Sycl variant id = " << vid
              << std::endl;
  }

}


RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MAT_MAT_SHARED, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
