//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

//
// Define work-group shape for SYCL execution
//
#define m_wg_sz (32)
#define g_wg_sz (integer::greater_of_squarest_factor_pair(work_group_size/m_wg_sz))
#define z_wg_sz (integer::lesser_of_squarest_factor_pair(work_group_size/m_wg_sz))

template <size_t work_group_size >
void LTIMES_NOVIEW::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> ndrange_dim(RAJA_DIVIDE_CEILING_INT(num_z, z_wg_sz),
                               RAJA_DIVIDE_CEILING_INT(num_g, g_wg_sz),
                               RAJA_DIVIDE_CEILING_INT(num_m, m_wg_sz));
    sycl::range<3> wkgroup_dim(z_wg_sz, g_wg_sz, m_wg_sz);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3> ( ndrange_dim * wkgroup_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          Index_type m = item.get_global_id(2);
          Index_type g = item.get_global_id(1);
          Index_type z = item.get_global_id(0);

          if (m < num_m && g < num_g && z < num_z) {
            for (Index_type d = 0; d < num_d; ++d) {
              LTIMES_NOVIEW_BODY;
            } 
          }

        });
      });
    }
    qu->wait();
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernel<
          RAJA::statement::For<1, RAJA::sycl_global_2<z_wg_sz>,      //z
            RAJA::statement::For<2, RAJA::sycl_global_1<g_wg_sz>,    //g
              RAJA::statement::For<3, RAJA::sycl_global_0<m_wg_sz>,  //m
                RAJA::statement::For<0, RAJA::seq_exec,              //d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                                               RAJA::RangeSegment(0, num_z),
                                               RAJA::RangeSegment(0, num_g),
                                               RAJA::RangeSegment(0, num_m)),
        [=] (Index_type d, Index_type z, Index_type g, Index_type m) {
        LTIMES_NOVIEW_BODY;
      });

    }
    qu->wait();
    stopTimer();

  } else {
     std::cout << "\n LTIMES_NOVIEW : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(LTIMES_NOVIEW, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
