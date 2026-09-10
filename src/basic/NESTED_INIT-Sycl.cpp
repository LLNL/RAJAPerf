//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
#define i_wg_sz (32)
#define j_wg_sz (work_group_size / i_wg_sz)
#define k_wg_sz (1)

template <size_t work_group_size >
void NESTED_INIT::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> global_dim(k_wg_sz * RAJA_DIVIDE_CEILING_INT(nk, k_wg_sz),
                              j_wg_sz * RAJA_DIVIDE_CEILING_INT(nj, j_wg_sz),
                              i_wg_sz * RAJA_DIVIDE_CEILING_INT(ni, i_wg_sz));
    sycl::range<3> wkgroup_dim(k_wg_sz, j_wg_sz, i_wg_sz);
  
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      qu.submit([&] (::sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3> ( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) { 

          Index_type i = item.get_global_id(2);
          Index_type j = item.get_global_id(1);
          Index_type k = item.get_global_id(0);

          if (i < ni && j < nj && k < nk) {
            NESTED_INIT_BODY;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();
  
  } else if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<2, RAJA::sycl_global_0<k_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_1<j_wg_sz>,
              RAJA::statement::For<0, RAJA::sycl_global_2<i_wg_sz>,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                         RAJA::RangeSegment(0, nj),
                         RAJA::RangeSegment(0, nk)),
        res,
        [=] (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(NESTED_INIT, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
