//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

using namespace ltimes_idx;

//
// Define work-group shape for SYCL execution
//
#define m_wg_sz (32)
#define g_wg_sz (integer::greater_of_squarest_factor_pair(work_group_size/m_wg_sz))
#define z_wg_sz (integer::lesser_of_squarest_factor_pair(work_group_size/m_wg_sz))

template <size_t work_group_size, size_t tune_idx >
void LTIMES::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  LTIMES_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<3> global_dim(z_wg_sz * RAJA_DIVIDE_CEILING_INT(*num_z, z_wg_sz),
                              g_wg_sz * RAJA_DIVIDE_CEILING_INT(*num_g, g_wg_sz),
                              m_wg_sz * RAJA_DIVIDE_CEILING_INT(*num_m, m_wg_sz));
    sycl::range<3> wkgroup_dim(z_wg_sz, g_wg_sz, m_wg_sz);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      qu.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3> ( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          IM m(item.get_global_id(2));
          IG g(item.get_global_id(1));
          IZ z(item.get_global_id(0));

          if (m < num_m && g < num_g && z < num_z) {
            for (ID d(0); d < num_d; ++d) {
              LTIMES_BODY;
            }
          }

        });
      });
      RP_CALI_SUBKERNEL_END("LTIMES_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    if constexpr (tune_idx == 0) {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::SyclKernelAsync<
            RAJA::statement::For<1, RAJA::sycl_global_0<z_wg_sz>,      //z
              RAJA::statement::For<2, RAJA::sycl_global_1<g_wg_sz>,    //g
                RAJA::statement::For<3, RAJA::sycl_global_2<m_wg_sz>,  //m
                  RAJA::statement::For<0, RAJA::seq_exec,              //d
                    RAJA::statement::Lambda<0>
                  >
                >
              >
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
        RAJA::kernel_resource<EXEC_POL>( 
          RAJA::make_tuple(IDRange(0, *num_d),
                           IZRange(0, *num_z),
                           IGRange(0, *num_g),
                           IMRange(0, *num_m)),
          res,
          [=] (ID d, IZ z, IG g, IM m) {
          LTIMES_BODY;
        });
        RP_CALI_SUBKERNEL_END("LTIMES_1");

      }
      stopTimer();

    } else if constexpr (tune_idx == 1) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

      using z_policy = RAJA::LoopPolicy<RAJA::sycl_global_item_0>;

      using g_policy = RAJA::LoopPolicy<RAJA::sycl_global_item_1>;

      using m_policy = RAJA::LoopPolicy<RAJA::sycl_global_item_2>;

      using d_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

      const size_t z_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_z, z_wg_sz);

      const size_t g_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_g, g_wg_sz);

      const size_t m_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_m, m_wg_sz);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(RAJA::Teams(m_grid_sz, g_grid_sz, z_grid_sz),
                               RAJA::Threads(m_wg_sz, g_wg_sz, z_wg_sz)),
            [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

              RAJA::loop<z_policy>(ctx, IZRange(0, *num_z),
                [&](IZ z) {
                  RAJA::loop<g_policy>(ctx, IGRange(0, *num_g),
                    [&](IG g) {
                      RAJA::loop<m_policy>(ctx, IMRange(0, *num_m),
                        [&](IM m) {
                          RAJA::loop<d_policy>(ctx, IDRange(0, *num_d),
                            [&](ID d) {
                              LTIMES_BODY
                            }
                          ); // RAJA::loop<d_policy>
                        }
                      ); // RAJA::loop<m_policy>
                    }
                  ); // RAJA::loop<g_policy>
                }
              ); // RAJA::loop<z_policy>

            } // outer lambda (ctx)
        );    // RAJA::launch
        RP_CALI_SUBKERNEL_END("LTIMES_1");

      } // loop over kernel reps
      stopTimer();
    }

  } else {
     std::cout << "\n LTIMES : Unknown Sycl variant id = " << vid << std::endl;
  }
}


void LTIMES::defineSyclVariantTunings()
{

  for (VariantID vid : {Base_SYCL, RAJA_SYCL}) {

    seq_for(gpu_block_sizes_type{}, [&](auto work_group_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(work_group_size)) {

        if (vid == RAJA_SYCL) {

          addVariantTuning<&LTIMES::runSyclVariantImpl<work_group_size, 0>>(
              vid, "kernel_"+std::to_string(work_group_size));

          addVariantTuning<&LTIMES::runSyclVariantImpl<work_group_size, 1>>(
              vid, "launch_"+std::to_string(work_group_size));

        } else {

          addVariantTuning<&LTIMES::runSyclVariantImpl<work_group_size, 0>>(
              vid, "block_"+std::to_string(work_group_size));

        }

      }

    });

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
