//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

template <size_t work_group_size >
void GEN_LIN_RECUR::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size1 = work_group_size * RAJA_DIVIDE_CEILING_INT(N, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size1, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type k = item.get_global_id(0);
          if (k < N) {
            GEN_LIN_RECUR_BODY1;
          }
 
        });
      });

      const size_t global_size2 = work_group_size * RAJA_DIVIDE_CEILING_INT(N+1, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size2, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i > 0 && i < N+1) {
            GEN_LIN_RECUR_BODY2;
          }

        });
      });
    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         RAJA::RangeSegment(0, N), [=] (Index_type k) {
         GEN_LIN_RECUR_BODY1;
       });

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         RAJA::RangeSegment(1, N+1), [=] (Index_type i) {
         GEN_LIN_RECUR_BODY2;
       });

    }
    stopTimer();

  } else {
     std::cout << "\n  GEN_LIN_RECUR : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, Sycl)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
