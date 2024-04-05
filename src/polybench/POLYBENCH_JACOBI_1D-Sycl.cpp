//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t work_group_size >
void POLYBENCH_JACOBI_1D::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        const size_t grid_size = work_group_size * RAJA_DIVIDE_CEILING_INT(N, work_group_size);

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                         [=] (sycl::nd_item<1> item) {

            Index_type i = item.get_global_id(0);
            if (i > 0 && i < N-1) {
              POLYBENCH_JACOBI_1D_BODY1;
            }

          });
        });

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                         [=] (sycl::nd_item<1> item) {

            Index_type i = item.get_global_id(0);
            if (i > 0 && i < N-1) {
              POLYBENCH_JACOBI_1D_BODY2;
            }

          });
        });

      }

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type t = 0; t < tsteps; ++t) {

        RAJA::forall< RAJA::sycl_exec<work_group_size, async>>(
           RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY1;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async>>(
           RAJA::RangeSegment{1, N-1}, [=] (Index_type i) {
            POLYBENCH_JACOBI_1D_BODY2;
        });

      }

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, Sycl)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

