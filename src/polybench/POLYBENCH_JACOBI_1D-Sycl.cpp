//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(N, work_group_size);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i > 0 && i < N-1) {
            POLYBENCH_JACOBI_1D_BODY1;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i > 0 && i < N-1) {
            POLYBENCH_JACOBI_1D_BODY2;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    using EXEC_POL = RAJA::sycl_exec<work_group_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY1;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
      RAJA::forall<EXEC_POL> ( res, RAJA::RangeSegment{1, N-1},
        [=] (Index_type i) {
          POLYBENCH_JACOBI_1D_BODY2;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
