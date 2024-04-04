//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

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
void LTIMES::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  LTIMES_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> global_dim(z_wg_sz * RAJA_DIVIDE_CEILING_INT(num_z, z_wg_sz),
                              g_wg_sz * RAJA_DIVIDE_CEILING_INT(num_g, g_wg_sz),
                              m_wg_sz * RAJA_DIVIDE_CEILING_INT(num_m, m_wg_sz));
    sycl::range<3> wkgroup_dim(z_wg_sz, g_wg_sz, m_wg_sz);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3> ( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          Index_type m = item.get_global_id(2);
          Index_type g = item.get_global_id(1);
          Index_type z = item.get_global_id(0);

          if (m < num_m && g < num_g && z < num_z) {
            for (Index_type d = 0; d < num_d; ++d) {
              LTIMES_BODY;
            }
          }

        });
      });
    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    LTIMES_VIEWS_RANGES_RAJA;

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

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                                 IZRange(0, num_z),
                                                 IGRange(0, num_g),
                                                 IMRange(0, num_m)),
          [=] (ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

  } else {
     std::cout << "\n LTIMES : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(LTIMES, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
