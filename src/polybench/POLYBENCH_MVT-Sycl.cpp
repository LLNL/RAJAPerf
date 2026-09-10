//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{


template < size_t work_group_size >
void POLYBENCH_MVT::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_MVT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(N, work_group_size);

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);

          if (i < N) {
            POLYBENCH_MVT_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY2;
            }
            POLYBENCH_MVT_BODY3;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
      qu.submit([&] (sycl::handler& h) { 
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) { 

          Index_type i = item.get_global_id(0);

          if (i < N) {
            POLYBENCH_MVT_BODY4;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY5;
            }
            POLYBENCH_MVT_BODY6;
          } 

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_MVT_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_0<work_group_size>,
            RAJA::statement::Lambda<0, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_1");
        RAJA::kernel_param_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},
          res,

          [=] (Real_type &dot) {
            POLYBENCH_MVT_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY2_RAJA;
          },
          [=] (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY3_RAJA;
          }

        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_MVT_2");
        RAJA::kernel_param_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          RAJA::tuple<Real_type>{0.0},
          res,

          [=] (Real_type &dot) {
            POLYBENCH_MVT_BODY4_RAJA;
          },
          [=] (Index_type i, Index_type j, Real_type &dot) {
            POLYBENCH_MVT_BODY5_RAJA;
          },
          [=] (Index_type i, Real_type &dot) {
            POLYBENCH_MVT_BODY6_RAJA;
          }

        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_MVT_2");

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_MVT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_MVT, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

