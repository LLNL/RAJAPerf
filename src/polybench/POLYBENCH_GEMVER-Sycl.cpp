//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

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
void POLYBENCH_GEMVER::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_GEMVER_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> global_dim1(1,
                               i_wg_sz * RAJA_DIVIDE_CEILING_INT(n, i_wg_sz),
                               j_wg_sz * RAJA_DIVIDE_CEILING_INT(n, j_wg_sz));
    sycl::range<3> wkgroup_dim1(1, i_wg_sz, j_wg_sz);

    const size_t global_size234 = work_group_size * RAJA_DIVIDE_CEILING_INT(n, work_group_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>( global_dim1, wkgroup_dim1),
                       [=] (sycl::nd_item<3> item) {

          Index_type i = item.get_global_id(1);
          Index_type j = item.get_global_id(2);

          if (i < n && j < n) {
            POLYBENCH_GEMVER_BODY1;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size234, work_group_size),
                       [=] (sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);
          if (i < n) {
            POLYBENCH_GEMVER_BODY2;
            for (Index_type j = 0; j < n; ++j) {
              POLYBENCH_GEMVER_BODY3;
            }
            POLYBENCH_GEMVER_BODY4;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size234, work_group_size),
                       [=] (sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);
          if (i < n) {
            POLYBENCH_GEMVER_BODY5;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(global_size234, work_group_size),
                       [=] (sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);
          if (i < n) {
            POLYBENCH_GEMVER_BODY6;
            for (Index_type j = 0; j < n; ++j) {
              POLYBENCH_GEMVER_BODY7;
            }
            POLYBENCH_GEMVER_BODY8;
          }

        });
      });
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_GEMVER_VIEWS_RAJA;

    using EXEC_POL1 =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<i_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<j_wg_sz>,
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    using EXEC_POL24 =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_0<work_group_size>,
            RAJA::statement::Lambda<0, RAJA::Segs<0>, RAJA::Params<0>>,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>, RAJA::Params<0>>
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0>, RAJA::Params<0>>
          >
        >
      >;

    using EXEC_POL3 = RAJA::sycl_exec<work_group_size, true /*async*/>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_1");
      RAJA::kernel_resource<EXEC_POL1>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        res,
        [=] (Index_type i, Index_type j) {
          POLYBENCH_GEMVER_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_2");
      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type /* i */, Real_type &dot) {
          POLYBENCH_GEMVER_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY3_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY4_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_2");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_3");
      RAJA::forall<EXEC_POL3> ( res, RAJA::RangeSegment{0, n},
        [=] (Index_type i) {
          POLYBENCH_GEMVER_BODY5_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_3");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_GEMVER_4");
      RAJA::kernel_param_resource<EXEC_POL24>(
        RAJA::make_tuple(RAJA::RangeSegment{0, n},
                         RAJA::RangeSegment{0, n}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY6_RAJA;
        },
        [=] (Index_type i, Index_type j, Real_type &dot) {
          POLYBENCH_GEMVER_BODY7_RAJA;
        },
        [=] (Index_type i, Real_type &dot) {
          POLYBENCH_GEMVER_BODY8_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_GEMVER_4");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GEMVER : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMVER, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

