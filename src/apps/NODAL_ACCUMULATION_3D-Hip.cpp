//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

#define NODAL_ACCUMULATION_3D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, m_nodal_array_length); \
  allocAndInitHipDeviceData(vol, m_vol, m_zonal_array_length); \
  allocAndInitHipDeviceData(real_zones, m_domain->real_zones, iend);

#define NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x, x, m_nodal_array_length); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(vol); \
  deallocHipDeviceData(real_zones);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void nodal_accumulation_3d(Real_ptr vol,
                      Real_ptr x0, Real_ptr x1,
                      Real_ptr x2, Real_ptr x3,
                      Real_ptr x4, Real_ptr x5,
                      Real_ptr x6, Real_ptr x7,
                      Index_ptr real_zones,
                      Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   if (ii < iend) {
     NODAL_ACCUMULATION_3D_BODY_INDEX;
     NODAL_ACCUMULATION_3D_BODY_ATOMIC(::atomicAdd);
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void nodal_accumulation_3d_unsafe(Real_ptr vol,
                      Real_ptr x0, Real_ptr x1,
                      Real_ptr x2, Real_ptr x3,
                      Real_ptr x4, Real_ptr x5,
                      Real_ptr x6, Real_ptr x7,
                      Index_ptr real_zones,
                      Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   if (ii < iend) {
     NODAL_ACCUMULATION_3D_BODY_INDEX;
     NODAL_ACCUMULATION_3D_BODY_ATOMIC(RAJAPERF_HIP_unsafeAtomicAdd);
   }
}


template < size_t block_size >
void NODAL_ACCUMULATION_3D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_HIP;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((nodal_accumulation_3d<block_size>), dim3(grid_size), dim3(block_size), 0, 0, vol,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       real_zones,
                                       iend);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_HIP;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto nodal_accumulation_3d_lambda = [=] __device__ (Index_type ii) {
        NODAL_ACCUMULATION_3D_BODY_INDEX;
        NODAL_ACCUMULATION_3D_BODY_ATOMIC(::atomicAdd);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(nodal_accumulation_3d_lambda)>),
                                       dim3(grid_size), dim3(block_size), 0, 0,
                                       ibegin, iend,
                                       nodal_accumulation_3d_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_HIP;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    camp::resources::Resource working_res{camp::resources::Hip()};
    RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                             m_domain->n_real_zones,
                                             working_res);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        zones, [=] __device__ (Index_type zone) {
          NODAL_ACCUMULATION_3D_BODY_ATOMIC(RAJA::atomicAdd<RAJA::hip_atomic>);
      });

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void NODAL_ACCUMULATION_3D::runHipVariantUnsafe(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_HIP;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((nodal_accumulation_3d_unsafe<block_size>), dim3(grid_size), dim3(block_size), 0, 0, vol,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       real_zones,
                                       iend);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_HIP;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto nodal_accumulation_3d_lambda = [=] __device__ (Index_type ii) {
        NODAL_ACCUMULATION_3D_BODY_INDEX;
        NODAL_ACCUMULATION_3D_BODY_ATOMIC(RAJAPERF_HIP_unsafeAtomicAdd);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(nodal_accumulation_3d_lambda)>),
                                       dim3(grid_size), dim3(block_size), 0, 0,
                                       ibegin, iend,
                                       nodal_accumulation_3d_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

void NODAL_ACCUMULATION_3D::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();

  size_t t = 0;

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runHipVariantImpl<block_size>(vid);

      }

      t += 1;

    }

  });

  if ( vid == Base_HIP || vid == Lambda_HIP ) {

    if (have_unsafe_atomics) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          if (tune_idx == t) {

            runHipVariantUnsafe<block_size>(vid);

          }

          t += 1;

        }

      });

    }

  }

}

void NODAL_ACCUMULATION_3D::setHipTuningDefinitions(VariantID vid)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

    }

  });

  if ( vid == Base_HIP || vid == Lambda_HIP ) {

    if (have_unsafe_atomics) {

      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {

          addVariantTuningName(vid, "unsafe_"+std::to_string(block_size));

        }

      });

    }

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
