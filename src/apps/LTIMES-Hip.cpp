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

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

using namespace ltimes_idx;

//
// Define thread block shape for Hip execution
//
#define LTIMES_M_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(static_cast<size_t>(*num_m), 1, 1);

#define LTIMES_M_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(*num_z), \
               static_cast<size_t>(*num_g), \
               1);

#define zgm_m_block_sz (32)
#define zgm_g_block_sz (integer::greater_of_squarest_factor_pair(block_size/zgm_m_block_sz))
#define zgm_z_block_sz (integer::lesser_of_squarest_factor_pair(block_size/zgm_m_block_sz))

#define LTIMES_ZGM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  zgm_m_block_sz, zgm_g_block_sz, zgm_z_block_sz

#define LTIMES_ZGM_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(LTIMES_ZGM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP);

#define LTIMES_ZGM_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(*num_m, zgm_m_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(*num_g, zgm_g_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(*num_z, zgm_z_block_sz)));


__global__ void ltimes_block_moments(PHI_VIEW phi, ELL_VIEW ell, PSI_VIEW psi,
                       ID num_d, IM num_m, IG num_g, IZ num_z)
{
   IG g(blockIdx.y);
   IZ z(blockIdx.x);

   if (g < num_g && z < num_z) {
     for (IM m(threadIdx.x); m < num_m; m += blockDim.x) {
       for (ID d(0); d < num_d; ++d ) {
         LTIMES_BODY;
       }
     }
   }
}

template < size_t m_block_size, size_t g_block_size, size_t z_block_size >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes_factorized(PHI_VIEW phi, ELL_VIEW ell, PSI_VIEW psi,
                           ID num_d, IM num_m, IG num_g, IZ num_z)
{
   IM m(blockIdx.x * m_block_size + threadIdx.x);
   IG g(blockIdx.y * g_block_size + threadIdx.y);
   IZ z(blockIdx.z * z_block_size + threadIdx.z);

   if (m < num_m && g < num_g && z < num_z) {
     for (ID d(0); d < num_d; ++d ) {
       LTIMES_BODY;
     }
   }
}

template < typename Lambda >
__global__ void ltimes_lam_block_moments(IM num_m, IG num_g, IZ num_z,
                           Lambda body)
{
   IG g(blockIdx.y);
   IZ z(blockIdx.x);

   if (g < num_g && z < num_z) {
     for (IM m(threadIdx.x); m < num_m; m += blockDim.x) {
       body(z, g, m);
     }
   }
}

template < size_t m_block_size, size_t g_block_size, size_t z_block_size, typename Lambda >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes_lam_factorized(IM num_m, IG num_g, IZ num_z,
                               Lambda body)
{
   IM m(blockIdx.x * m_block_size + threadIdx.x);
   IG g(blockIdx.y * g_block_size + threadIdx.y);
   IZ z(blockIdx.z * z_block_size + threadIdx.z);

   if (m < num_m && g < num_g && z < num_z) {
     body(z, g, m);
   }
}


template < size_t tune_idx, size_t block_size >
void LTIMES::runHipVariantImpl(VariantID vid)
{
  if constexpr (tune_idx == 0 || tune_idx == 2) {
    setBlockSize(m_num_m);
  } else {
    setBlockSize(block_size);
  }

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  LTIMES_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
      constexpr size_t shmem = 0;

      if constexpr (tune_idx == 0) {
        LTIMES_M_THREADS_PER_BLOCK_HIP;
        LTIMES_M_NBLOCKS_HIP;

        RPlaunchHipKernel(
          (ltimes_block_moments),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          phi, ell, psi,
          num_d, num_m, num_g, num_z );
      } else if constexpr (tune_idx == 1) {
        LTIMES_ZGM_THREADS_PER_BLOCK_HIP;
        LTIMES_ZGM_NBLOCKS_HIP;

        RPlaunchHipKernel(
          (ltimes_factorized<LTIMES_ZGM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          phi, ell, psi,
          num_d, num_m, num_g, num_z );
      }
      RP_CALI_SUBKERNEL_END("LTIMES_1");
    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
      RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");

      auto ltimes_lambda = [=] __device__ (IZ z, IG g, IM m) {
       for (ID d(0); d < num_d; ++d ) {
         LTIMES_BODY;
       }
      };

      constexpr size_t shmem = 0;

      if constexpr (tune_idx == 0) {
        LTIMES_M_THREADS_PER_BLOCK_HIP;
        LTIMES_M_NBLOCKS_HIP;

        RPlaunchHipKernel(
          (ltimes_lam_block_moments<decltype(ltimes_lambda)>),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          num_m, num_g, num_z,
          ltimes_lambda );
      } else if constexpr (tune_idx == 1) {
        LTIMES_ZGM_THREADS_PER_BLOCK_HIP;
        LTIMES_ZGM_NBLOCKS_HIP;

        RPlaunchHipKernel(
          (ltimes_lam_factorized<LTIMES_ZGM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                          decltype(ltimes_lambda)>),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          num_m, num_g, num_z,
          ltimes_lambda );
      }
      RP_CALI_SUBKERNEL_END("LTIMES_1");
    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    if constexpr (tune_idx == 0) {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::HipKernelAsync<
            RAJA::statement::For<1, RAJA::hip_block_x_loop, // z
              RAJA::statement::For<2, RAJA::hip_block_y_loop, // g
                RAJA::statement::For<3, RAJA::hip_thread_x_loop, // m
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
        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(IDRange(0, *num_d),
                           IZRange(0, *num_z),
                           IGRange(0, *num_g),
                           IMRange(0, *num_m)),
          res,
          [=] __device__ (ID d, IZ z, IG g, IM m) {
            LTIMES_BODY;
          }
        );
        RP_CALI_SUBKERNEL_END("LTIMES_1");
      }
      stopTimer();

    } else if constexpr (tune_idx == 1) {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::HipKernelFixedAsync<zgm_m_block_sz*zgm_g_block_sz*zgm_z_block_sz,
            RAJA::statement::For<1, RAJA::hip_global_size_z_direct<zgm_z_block_sz>,     // z
              RAJA::statement::For<2, RAJA::hip_global_size_y_direct<zgm_g_block_sz>,   // g
                RAJA::statement::For<3, RAJA::hip_global_size_x_direct<zgm_m_block_sz>, // m
                  RAJA::statement::For<0, RAJA::seq_exec,                               // d
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
          [=] __device__ (ID d, IZ z, IG g, IM m) {
            LTIMES_BODY;
          }
        );
        RP_CALI_SUBKERNEL_END("LTIMES_1");
      }
      stopTimer();

    } else if constexpr (tune_idx == 2) {

      constexpr bool async = true;

      using launch_policy =
          RAJA::LaunchPolicy<RAJA::hip_launch_t<async>>;

      using z_policy = RAJA::LoopPolicy<RAJA::hip_block_x_loop>;

      using g_policy = RAJA::LoopPolicy<RAJA::hip_block_y_loop>;

      using m_policy = RAJA::LoopPolicy<RAJA::hip_thread_x_loop>;

      using d_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");

        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(RAJA::Teams(*num_z, *num_g, 1),
                               RAJA::Threads(*num_m, 1, 1)),
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
    } else if constexpr (tune_idx == 3) {

      constexpr bool async = true;

      using launch_policy =
          RAJA::LaunchPolicy<RAJA::hip_launch_t<async, zgm_m_block_sz*zgm_g_block_sz*zgm_z_block_sz>>;

      using z_policy = RAJA::LoopPolicy<RAJA::hip_global_size_z_direct<zgm_z_block_sz>>;

      using g_policy = RAJA::LoopPolicy<RAJA::hip_global_size_y_direct<zgm_g_block_sz>>;

      using m_policy = RAJA::LoopPolicy<RAJA::hip_global_size_x_direct<zgm_m_block_sz>>;

      using d_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

      const size_t z_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_z, zgm_z_block_sz);

      const size_t g_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_g, zgm_g_block_sz);

      const size_t m_grid_sz = RAJA_DIVIDE_CEILING_INT(*num_m, zgm_m_block_sz);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {
        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");

        RAJA::launch<launch_policy>( res,
            RAJA::LaunchParams(RAJA::Teams(m_grid_sz, g_grid_sz, z_grid_sz),
                               RAJA::Threads(zgm_m_block_sz, zgm_g_block_sz, zgm_z_block_sz)),
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
     getCout() << "\n LTIMES : Unknown Hip variant id = " << vid << std::endl;
  }
}


template < size_t tune_idx >
void LTIMES::runHipVariantM(VariantID vid)
{
  runHipVariantImpl<tune_idx>(vid);
}

template < size_t tune_idx, size_t block_size >
void LTIMES::runHipVariantZGM(VariantID vid)
{
  runHipVariantImpl<tune_idx, block_size>(vid);
}


void LTIMES::defineHipVariantTunings()
{

  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {

    const size_t m_block_size = static_cast<size_t>(m_num_m);

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(m_block_size)) {

      if (vid == RAJA_HIP) {
        addVariantTuning<&LTIMES::runHipVariantM<0>>(
            vid, "kernel_m_"+std::to_string(m_block_size));
        addVariantTuning<&LTIMES::runHipVariantM<2>>(
            vid, "launch_m_"+std::to_string(m_block_size));
      } else {
        addVariantTuning<&LTIMES::runHipVariantM<0>>(
            vid, "block_m_"+std::to_string(m_block_size));
      }

    }

  }

  for (VariantID vid : {Base_HIP, Lambda_HIP, RAJA_HIP}) {

    seq_for(zgm_gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (vid == RAJA_HIP) {
          addVariantTuning<&LTIMES::runHipVariantZGM<1, block_size>>(
              vid, "kernel_zgm_"+std::to_string(block_size));
          addVariantTuning<&LTIMES::runHipVariantZGM<3, block_size>>(
              vid, "launch_zgm_"+std::to_string(block_size));
        } else {
          addVariantTuning<&LTIMES::runHipVariantZGM<1, block_size>>(
              vid, "block_zgm_"+std::to_string(block_size));
        }

      }

    });

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
