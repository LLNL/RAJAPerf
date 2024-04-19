//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace polybench
{

   //
   // Define work-group shape for SYCL execution
   //
#define j_wg_sz (32)
#define i_wg_sz (work_group_size / j_wg_sz)


template < size_t work_group_size >
void POLYBENCH_FLOYD_WARSHALL::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<2> global_dim(i_wg_sz * RAJA_DIVIDE_CEILING_INT(N, i_wg_sz),
                              j_wg_sz * RAJA_DIVIDE_CEILING_INT(N, j_wg_sz));

    sycl::range<2> wkgroup_dim(i_wg_sz, j_wg_sz);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type k = 0; k < N; ++k) {

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<2>( global_dim, wkgroup_dim),
                         [=] (sycl::nd_item<2> item) {

            Index_type i = item.get_global_id(0);
            Index_type j = item.get_global_id(1);

            if ( i < N && j < N ) {
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }

          });
        });

      }

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
#if 0
          RAJA::statement::SyclKernelAsync<
#else
          RAJA::statement::SyclKernel<
#endif
            RAJA::statement::For<1, RAJA::sycl_global_0<i_wg_sz>,
              RAJA::statement::For<2, RAJA::sycl_global_1<j_wg_sz>,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        res,
        [=] (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FLOYD_WARSHALL, Sycl)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

