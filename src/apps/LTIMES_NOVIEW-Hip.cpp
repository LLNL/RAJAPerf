//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Define thread block shape for Hip execution
//
#define m_block_sz (32)
#define g_block_sz (integer::greater_of_squarest_factor_pair(block_size/m_block_sz))
#define z_block_sz (integer::lesser_of_squarest_factor_pair(block_size/m_block_sz))

#define LTIMES_NOVIEW_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  m_block_sz, g_block_sz, z_block_sz

#define LTIMES_NOVIEW_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(LTIMES_NOVIEW_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP);

#define LTIMES_NOVIEW_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_m, m_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_g, g_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_z, z_block_sz)));


template < size_t m_block_size, size_t g_block_size, size_t z_block_size >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes_noview(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                              Index_type num_d,
                              Index_type num_m, Index_type num_g, Index_type num_z)
{
   Index_type m = blockIdx.x * m_block_size + threadIdx.x;
   Index_type g = blockIdx.y * g_block_size + threadIdx.y;
   Index_type z = blockIdx.z * z_block_size + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     for (Index_type d = 0; d < num_d; ++d ) {
       LTIMES_NOVIEW_BODY;
     }
   }
}

template < size_t m_block_size, size_t g_block_size, size_t z_block_size, typename Lambda >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes_noview_lam(Index_type num_m, Index_type num_g, Index_type num_z,
                                  Lambda body)
{
   Index_type m = blockIdx.x * m_block_size + threadIdx.x;
   Index_type g = blockIdx.y * g_block_size + threadIdx.y;
   Index_type z = blockIdx.z * z_block_size + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     body(z, g, m);
   }
}


template < size_t block_size, size_t tune_idx >
void LTIMES_NOVIEW::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  LTIMES_NOVIEW_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
      LTIMES_NOVIEW_THREADS_PER_BLOCK_HIP;
      LTIMES_NOVIEW_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RPlaunchHipKernel(
        (ltimes_noview<LTIMES_NOVIEW_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        phidat, elldat, psidat,
        num_d, num_m, num_g, num_z );
      RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
      auto ltimes_noview_lambda = 
        [=] __device__ (Index_type z, Index_type g, Index_type m) {
          for (Index_type d = 0; d < num_d; ++d ) {
            LTIMES_NOVIEW_BODY;
          }
        };

      LTIMES_NOVIEW_THREADS_PER_BLOCK_HIP;
      LTIMES_NOVIEW_NBLOCKS_HIP;
      constexpr size_t shmem = 0;

      RPlaunchHipKernel(
        (ltimes_noview_lam<LTIMES_NOVIEW_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                           decltype(ltimes_noview_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        num_m, num_g, num_z,
        ltimes_noview_lambda );
      RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    if constexpr (tune_idx == 0) {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::HipKernelFixedAsync<m_block_sz*g_block_sz*z_block_sz,
            RAJA::statement::For<1, RAJA::hip_global_size_z_direct<z_block_sz>,     //z
              RAJA::statement::For<2, RAJA::hip_global_size_y_direct<g_block_sz>,   //g
                RAJA::statement::For<3, RAJA::hip_global_size_x_direct<m_block_sz>, //m
                  RAJA::statement::For<0, RAJA::seq_exec,          //d
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

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                           RAJA::RangeSegment(0, num_z),
                           RAJA::RangeSegment(0, num_g),
                           RAJA::RangeSegment(0, num_m)),
          res,
          [=] __device__ (Index_type d, Index_type z,
                          Index_type g, Index_type m) {
            LTIMES_NOVIEW_BODY;
          }
        );
        RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

      }
      stopTimer();

    } else if constexpr (tune_idx == 1) {

      constexpr bool async = true;

      using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, m_block_sz*g_block_sz*z_block_sz>>;

      using z_policy = RAJA::LoopPolicy<RAJA::hip_global_size_z_loop<z_block_sz>>;

      using g_policy = RAJA::LoopPolicy<RAJA::hip_global_size_y_loop<g_block_sz>>;

      using m_policy = RAJA::LoopPolicy<RAJA::hip_global_size_x_loop<m_block_sz>>;

      using d_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

      const size_t z_grid_sz = RAJA_DIVIDE_CEILING_INT(num_z, z_block_sz);

      const size_t g_grid_sz = RAJA_DIVIDE_CEILING_INT(num_g, g_block_sz);

      const size_t m_grid_sz = RAJA_DIVIDE_CEILING_INT(num_m, m_block_sz);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_NOVIEW_1");
        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(RAJA::Teams(m_grid_sz, g_grid_sz, z_grid_sz),
                               RAJA::Threads(m_block_sz, g_block_sz, z_block_sz)),
            [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

              RAJA::loop<z_policy>(ctx, RAJA::RangeSegment(0, num_z),
                [&](Index_type z) {
                  RAJA::loop<g_policy>(ctx, RAJA::RangeSegment(0, num_g),
                    [&](Index_type g) {
                      RAJA::loop<m_policy>(ctx, RAJA::RangeSegment(0, num_m),
                        [&](Index_type m) {
                          RAJA::loop<d_policy>(ctx, RAJA::RangeSegment(0, num_d),
                            [&](Index_type d) {
                              LTIMES_NOVIEW_BODY
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
        RP_CALI_SUBKERNEL_END("LTIMES_NOVIEW_1");

      } // loop over kernel reps
      stopTimer();
    }

  } else {
     getCout() << "\n LTIMES_NOVIEW : Unknown Hip variant id = " << vid << std::endl;
  }
}


void LTIMES_NOVIEW::defineHipVariantTunings()
{

  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (vid == RAJA_HIP) {

          addVariantTuning<&LTIMES_NOVIEW::runHipVariantImpl<block_size, 0>>(
              vid, "kernel_"+std::to_string(block_size));

          addVariantTuning<&LTIMES_NOVIEW::runHipVariantImpl<block_size, 1>>(
              vid, "launch_"+std::to_string(block_size));

        } else {

          addVariantTuning<&LTIMES_NOVIEW::runHipVariantImpl<block_size, 0>>(
              vid, "block_"+std::to_string(block_size));

        }

      }

    });

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
