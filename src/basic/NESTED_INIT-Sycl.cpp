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

  //
  // Define work-group shape for SYCL execution
  //
#define i_block_sz (32)
#define j_block_sz (work_group_size / i_block_sz)
#define k_block_sz (1)

template <size_t work_group_size >
void NESTED_INIT::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    if (work_group_size > 0) {

      sycl::range<3> ndrange_dim(RAJA_DIVIDE_CEILING_INT(nk, k_block_sz),
                                 RAJA_DIVIDE_CEILING_INT(nj, j_block_sz),
                                 RAJA_DIVIDE_CEILING_INT(ni, i_block_sz));
      sycl::range<3> wkgroup_dim(k_block_sz, j_block_sz, i_block_sz);
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
        qu->submit([&] (cl::sycl::handler& h) {
          h.parallel_for(sycl::nd_range<3> ( ndrange_dim * wkgroup_dim, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type i = item.get_global_id(2);
            Index_type j = item.get_global_id(1);
            Index_type k = item.get_global_id(0);

            if (i < ni && j < nj && k < nk) {
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
          RAJA::statement::For<2, RAJA::sycl_global_0<k_block_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_1<j_block_sz>,
              RAJA::statement::For<0, RAJA::sycl_global_2<i_block_sz>,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
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
