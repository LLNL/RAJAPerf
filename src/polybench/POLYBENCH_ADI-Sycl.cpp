//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

template < size_t work_group_size >
void POLYBENCH_ADI::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_ADI_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(n-2, work_group_size);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ADI_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0) + 1;

          if (i < n-1) {
            POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
               POLYBENCH_ADI_BODY3;
            }
            POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
               POLYBENCH_ADI_BODY5;
            }
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_ADI_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ADI_2");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0) + 1;

          if (i < n-1) {
            POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
               POLYBENCH_ADI_BODY7;
            }
            POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
               POLYBENCH_ADI_BODY9;
            }
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_ADI_2");

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_ADI_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_2<work_group_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>>,
            RAJA::statement::For<2, RAJA::seq_exec,
              RAJA::statement::Lambda<3, RAJA::Segs<0,2>>
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ADI_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                         RAJA::RangeSegment{1, n-1},
                         RAJA::RangeStrideSegment{n-2, 0, -1}),
        res,

        [=] (Index_type i) {
          POLYBENCH_ADI_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j) {
          POLYBENCH_ADI_BODY3_RAJA;
        },
        [=] (Index_type i) {
          POLYBENCH_ADI_BODY4_RAJA;
        },
        [=] (Index_type i, Index_type k) {
          POLYBENCH_ADI_BODY5_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_ADI_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_ADI_2");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                         RAJA::RangeSegment{1, n-1},
                         RAJA::RangeStrideSegment{n-2, 0, -1}),
        res,

        [=] (Index_type i) {
          POLYBENCH_ADI_BODY6_RAJA;
        },
        [=] (Index_type i, Index_type j) {
          POLYBENCH_ADI_BODY7_RAJA;
        },
        [=] (Index_type i) {
          POLYBENCH_ADI_BODY8_RAJA;
        },
        [=] (Index_type i, Index_type k) {
          POLYBENCH_ADI_BODY9_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_ADI_2");

    } // run_reps
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_ADI : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_ADI, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

