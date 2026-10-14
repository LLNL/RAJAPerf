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

template < size_t work_group_size >
void NESTED_INIT::runSyclVariantFornest(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::fornest_collapsed_policy<RAJA::sycl_exec<work_group_size>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

template < size_t work_group_size, size_t tile_k, size_t tile_j, size_t tile_i >
void NESTED_INIT::runSyclVariantFornestRuntimeTiled(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::sycl_exec<work_group_size>,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime,
                                  RAJA::fornest_tile_runtime>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res,
                    EXEC_POL {RAJA::TileSize(tile_k),
                              RAJA::TileSize(tile_j),
                              RAJA::TileSize(tile_i)},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

template < size_t work_group_size >
void NESTED_INIT::runSyclVariantFornestAutoTiled(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};

  NESTED_INIT_DATA_SETUP;

  if ( vid == RAJA_SYCL ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::sycl_exec<work_group_size>,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("NESTED_INIT_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk), RAJA::range(nj), RAJA::range(ni),
                    [=] (Index_type k, Index_type j, Index_type i) {
        NESTED_INIT_BODY;
      });
      RP_CALI_SUBKERNEL_END("NESTED_INIT_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

void NESTED_INIT::defineSyclVariantTunings()
{
  for (VariantID vid : {Base_SYCL, RAJA_SYCL}) {

    seq_for(gpu_block_sizes_type{}, [&](auto work_group_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(work_group_size)) {

        addVariantTuning<&NESTED_INIT::runSyclVariantImpl<work_group_size>>(
            vid, "block_"+std::to_string(work_group_size));

        if (vid == RAJA_SYCL) {
          addVariantTuning<&NESTED_INIT::runSyclVariantFornest<work_group_size>>(
              vid, "fornest-collapse_block_"+std::to_string(work_group_size));
          addVariantTuning<&NESTED_INIT::runSyclVariantFornestAutoTiled<work_group_size>>(
              vid, "fornest-auto-tile_block_"+std::to_string(work_group_size));

          if constexpr (decltype(work_group_size)::value == 1 * 8 * 32) {
            addVariantTuning<&NESTED_INIT::runSyclVariantFornestRuntimeTiled<work_group_size, 1, 8, 32>>(
                vid, "fornest-runtime-tile_1x8x32_block_"+std::to_string(work_group_size));
          }
          if constexpr (decltype(work_group_size)::value == 2 * 4 * 32) {
            addVariantTuning<&NESTED_INIT::runSyclVariantFornestRuntimeTiled<work_group_size, 2, 4, 32>>(
                vid, "fornest-runtime-tile_2x4x32_block_"+std::to_string(work_group_size));
          }
          if constexpr (decltype(work_group_size)::value == 4 * 4 * 16) {
            addVariantTuning<&NESTED_INIT::runSyclVariantFornestRuntimeTiled<work_group_size, 4, 4, 16>>(
                vid, "fornest-runtime-tile_4x4x16_block_"+std::to_string(work_group_size));
          }
        }

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
