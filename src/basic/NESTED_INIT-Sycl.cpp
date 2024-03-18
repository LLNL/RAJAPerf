//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template <size_t work_group_size >
void NESTED_INIT::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    if (work_group_size > 0) {
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(ni, work_group_size);
  
        qu->submit([&] (cl::sycl::handler& h) {
          h.parallel_for(sycl::nd_range<3> (
                            sycl::range<3> (nk, nj, global_size),
                            sycl::range<3> (1, 1, work_group_size)),
                         [=] (sycl::nd_item<3> item) {

            Index_type i = item.get_global_id(2);
            Index_type j = item.get_global_id(1);
            Index_type k = item.get_global_id(0);

            if (i < ni) {
              NESTED_INIT_BODY
            }
          });
        });
  
      }
      qu->wait();
      stopTimer();
  
    } else {
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu->submit([&] (sycl::handler& h) {
          h.parallel_for(sycl::range<3> (nk, nj, ni),
                                           [=] (sycl::item<3> item) {

            Index_type i = item.get_id(2);
            Index_type j = item.get_id(1);
            Index_type k = item.get_id(0);

            NESTED_INIT_BODY
 
          });
        });
  
      }
      qu->wait();
      stopTimer();
  
    } 

  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  NESTED_INIT : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_2<work_group_size>,      // i
            RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
              RAJA::statement::For<2, RAJA::sycl_global_0<1>, // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, nk),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, ni)),
        [=] (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    qu->wait();
    stopTimer();

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(NESTED_INIT, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
