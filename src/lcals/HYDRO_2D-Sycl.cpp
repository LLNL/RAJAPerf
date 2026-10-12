//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

  //
  // Define work-group shape for SYCL execution
  //
#define j_wg_sz (32)
#define k_wg_sz (work_group_size / j_wg_sz)

template <size_t work_group_size >
void HYDRO_2D::runSyclVariantImpl(VariantID vid) {

  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> global_dim(1,
                              k_wg_sz * RAJA_DIVIDE_CEILING_INT(kn-2, k_wg_sz),
                              j_wg_sz * RAJA_DIVIDE_CEILING_INT(jn-2, j_wg_sz));
    sycl::range<3> wkgroup_dim(1, k_wg_sz, j_wg_sz);
 
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
      qu.submit([&] (sycl::handler& h) { 

        h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          int j = item.get_global_id(2) + 1;
          int k = item.get_global_id(1) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY1
          }

        });
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
      qu.submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          int j = item.get_global_id(2) + 1;
          int k = item.get_global_id(1) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY2
          }

        });
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
      qu.submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          int j = item.get_global_id(2) + 1;
          int k = item.get_global_id(1) + 1; 

          if (j < jn-1 && k < kn-1) {
            HYDRO_2D_BODY3
          }

        });
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    HYDRO_2D_VIEWS_RAJA;

    using EXECPOL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<k_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<j_wg_sz>,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_1");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res, 
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_2");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res, 
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("HYDRO_2D_3");
      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res, 
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });
      RP_CALI_SUBKERNEL_END("HYDRO_2D_3");

    }
    stopTimer();

  } else { 
     std::cout << "\n  HYDRO_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
