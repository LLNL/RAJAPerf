//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define work-group shape for SYCL execution
  //
#define k_wg_sz (32)
#define j_wg_sz (work_group_size / k_wg_sz)
#define i_wg_sz (1)


template < size_t work_group_size >
void POLYBENCH_HEAT_3D::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        sycl::range<3> global_dim(i_wg_sz * RAJA_DIVIDE_CEILING_INT(N-2, i_wg_sz),
                                  j_wg_sz * RAJA_DIVIDE_CEILING_INT(N-2, j_wg_sz),
                                  k_wg_sz * RAJA_DIVIDE_CEILING_INT(N-2, k_wg_sz));

        sycl::range<3> wkgroup_dim(i_wg_sz, j_wg_sz, k_wg_sz);

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type i = 1 + item.get_global_id(0);
            Index_type j = 1 + item.get_global_id(1);
            Index_type k = 1 + item.get_global_id(2);

            if (i < N-1 && j < N-1 && k < N-1) {
              POLYBENCH_HEAT_3D_BODY1;
            }

          });
        });

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type i = 1 + item.get_global_id(0);
            Index_type j = 1 + item.get_global_id(1);
            Index_type k = 1 + item.get_global_id(2);

            if (i < N-1 && j < N-1 && k < N-1) {
              POLYBENCH_HEAT_3D_BODY2;
            }

          });
        });

      }

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
#if 0
        RAJA::statement::SyclKernelAsync<
#else
        RAJA::statement::SyclKernel<
#endif
          RAJA::statement::For<0, RAJA::sycl_global_0<i_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_1<j_wg_sz>,
              RAJA::statement::For<2, RAJA::sycl_global_2<k_wg_sz>,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1}),
          res,
          [=] (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY1_RAJA;
          }
        );

        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1},
                           RAJA::RangeSegment{1, N-1}),
          res,
          [=] (Index_type i, Index_type j, Index_type k) {
            POLYBENCH_HEAT_3D_BODY2_RAJA;
          }
        );

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_HEAT_3D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_HEAT_3D, Sycl)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
