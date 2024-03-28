//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

#define j_block_sz (32)
#define k_block_sz (work_group_size / j_block_sz)

template <size_t work_group_size >
void HYDRO_2D::runSyclVariantImpl(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {
 
    auto kn_global_size = k_block_sz * RAJA_DIVIDE_CEILING_INT(kn-2, k_block_sz);
    auto jn_global_size = j_block_sz * RAJA_DIVIDE_CEILING_INT(jn-2, j_block_sz);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      qu->submit([&] (sycl::handler& h) { 

        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(kn_global_size, jn_global_size),
                                         sycl::range<2>(k_block_sz,j_block_sz)),
                       [=] (sycl::nd_item<2> item) {

          int j = item.get_global_id(1) + 1;
          int k = item.get_global_id(0) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY1
          }

        });
      });

      qu->submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(kn_global_size, jn_global_size),
                                         sycl::range<2>(k_block_sz,j_block_sz)),
                       [=] (sycl::nd_item<2> item) {

          int j = item.get_global_id(1) + 1;
          int k = item.get_global_id(0) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY2
          }

        });
      });

      qu->submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(kn_global_size, jn_global_size),
                                         sycl::range<2>(k_block_sz,j_block_sz)),
                       [=] (sycl::nd_item<2> item) {

          int j = item.get_global_id(1) + 1;
          int k = item.get_global_id(0) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY3
          }

        });
      });

    }
    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    HYDRO_2D_VIEWS_RAJA;

    using EXECPOL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<k_block_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<j_block_sz>,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    qu->wait();
    stopTimer();

  } else { 
     std::cout << "\n  HYDRO_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, Sycl)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
