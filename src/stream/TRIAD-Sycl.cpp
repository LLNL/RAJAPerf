//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace stream
{

template <size_t work_group_size >
void TRIAD::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  TRIAD_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            TRIAD_BODY
          }

        });
      });
    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         TRIAD_BODY;
       });

    }
    stopTimer();

  } else {
      std::cout << "\n  TRIAD : Unknown Sycl variant id = " << vid << std::endl;
  }

}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIAD, Sycl)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
