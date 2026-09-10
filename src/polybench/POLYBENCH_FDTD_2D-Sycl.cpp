//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define work-group shape for SYCL execution
  //
#define j_wg_sz (32)
#define i_wg_sz (work_group_size / j_wg_sz)


template < size_t work_group_size >
void POLYBENCH_FDTD_2D::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_FDTD_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size1 = work_group_size * RAJA_DIVIDE_CEILING_INT(ny, work_group_size);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size1, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type j = item.get_global_id(0);
          if (j < ny) {
             POLYBENCH_FDTD_2D_BODY1;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");

      sycl::range<3> global_dim234(1,
                                   i_wg_sz * RAJA_DIVIDE_CEILING_INT(nx, i_wg_sz),
                                   j_wg_sz * RAJA_DIVIDE_CEILING_INT(ny, j_wg_sz));

      sycl::range<3> wkgroup_dim234(1, i_wg_sz, j_wg_sz);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>( global_dim234, wkgroup_dim234),
                       [=] (sycl::nd_item<3> item) {

          Index_type i = item.get_global_id(1);
          Index_type j = item.get_global_id(2);

          if (i > 0 && i < nx && j < ny) {
            POLYBENCH_FDTD_2D_BODY2;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>( global_dim234, wkgroup_dim234),
                       [=] (sycl::nd_item<3> item) {

          Index_type i = item.get_global_id(1);
          Index_type j = item.get_global_id(2);

          if (i < nx && j > 0 && j < ny) {
            POLYBENCH_FDTD_2D_BODY3;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>( global_dim234, wkgroup_dim234),
                       [=] (sycl::nd_item<3> item) {

          Index_type i = item.get_global_id(1);
          Index_type j = item.get_global_id(2);

          if (i < nx-1 && j < ny-1) {
            POLYBENCH_FDTD_2D_BODY4;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

      t = (t+1) % m_tsteps;
    } // run_reps
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_FDTD_2D_VIEWS_RAJA;

    using EXEC_POL1 = RAJA::sycl_exec<work_group_size, true /*async*/>;

    using EXEC_POL234 =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<i_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<j_wg_sz>,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_1");
      RAJA::forall<EXEC_POL1>( res, RAJA::RangeSegment(0, ny),
      [=] (Index_type j) {
        POLYBENCH_FDTD_2D_BODY1_RAJA;
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_2");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{1, nx},
                         RAJA::RangeSegment{0, ny}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_3");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nx},
                         RAJA::RangeSegment{1, ny}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY3_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FDTD_2D_4");
      RAJA::kernel_resource<EXEC_POL234>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nx-1},
                         RAJA::RangeSegment{0, ny-1}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_FDTD_2D_BODY4_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FDTD_2D_4");

      t = (t+1) % m_tsteps;
    } // run_reps
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_FDTD_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FDTD_2D, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

