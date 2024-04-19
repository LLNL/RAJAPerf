//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t work_group_size >
void POLYBENCH_ATAX::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue(); 

  POLYBENCH_ATAX_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = work_group_size * RAJA_DIVIDE_CEILING_INT(N, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);

          if (i < N) {
            POLYBENCH_ATAX_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_ATAX_BODY2;
            }
            POLYBENCH_ATAX_BODY3;
          }

        });
      });

      qu->submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) { 

          Index_type j = item.get_global_id(0);

          if (j < N) {
            POLYBENCH_ATAX_BODY4;
            for (Index_type i = 0; i < N; ++i ) {
              POLYBENCH_ATAX_BODY5;
            }
            POLYBENCH_ATAX_BODY6;
          }

        });
      });

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_ATAX_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
#if 0
        RAJA::statement::SyclKernelAsync<
#else
        RAJA::statement::SyclKernel<
#endif
          RAJA::statement::For<0, RAJA::sycl_global_0<work_group_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >
      >;

    using EXEC_POL2 =
      RAJA::KernelPolicy<
#if 0
        RAJA::statement::SyclKernelAsync<
#else
        RAJA::statement::SyclKernel<
#endif
          RAJA::statement::For<1, RAJA::sycl_global_0<work_group_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<1>, RAJA::Params<0>>,
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<1>, RAJA::Params<0>>
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},

        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY2_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_ATAX_BODY3_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL2>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        RAJA::tuple<Real_type>{0.0},

        [=] (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY4_RAJA;
        },
        [=] (Index_type i, Index_type j , Real_type &dot) {
          POLYBENCH_ATAX_BODY5_RAJA;
        },
        [=] (Index_type j, Real_type &dot) {
          POLYBENCH_ATAX_BODY6_RAJA;
        }

     );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_ATAX : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_ATAX, Sycl)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
