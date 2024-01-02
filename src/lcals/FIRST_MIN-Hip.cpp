//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace rajaperf
{
namespace lcals
{


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void first_min(Real_ptr x,
                          MyMinLoc* dminloc,
                          MyMinLoc mininit,
                          Index_type iend)
{
  extern __shared__ MyMinLoc minloc[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  minloc[ threadIdx.x ] = mininit;

  for ( ; i < iend ; i += gridDim.x * block_size ) {
    MyMinLoc& mymin = minloc[ threadIdx.x ];
    FIRST_MIN_BODY;
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      if ( minloc[ threadIdx.x + i].val < minloc[ threadIdx.x ].val ) {
        minloc[ threadIdx.x ] = minloc[ threadIdx.x + i];
      }
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    dminloc[blockIdx.x] = minloc[ 0 ];
  }
}


template < size_t block_size, typename MappingHelper >
void FIRST_MIN::runHipVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  FIRST_MIN_DATA_SETUP;

  if ( vid == Base_HIP ) {

    constexpr size_t shmem = sizeof(MyMinLoc)*block_size;
    const size_t max_grid_size = RAJAPERF_HIP_GET_MAX_BLOCKS(
        MappingHelper, (first_min<block_size>), block_size, shmem);

    const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const size_t grid_size = std::min(normal_grid_size, max_grid_size);

    RAJAPERF_HIP_REDUCER_SETUP(MyMinLoc*, dminloc, mymin_block, grid_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      FIRST_MIN_MINLOC_INIT;
      RAJAPERF_HIP_REDUCER_INITIALIZE_VALUE(mymin, dminloc, mymin_block, grid_size);

      RPlaunchHipKernel( (first_min<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         x, dminloc, mymin,
                         iend );

      RAJAPERF_HIP_REDUCER_COPY_BACK_NOFINAL(dminloc, mymin_block, grid_size);
      for (Index_type i = 0; i < static_cast<Index_type>(grid_size); i++) {
        if ( mymin_block[i].val < mymin.val ) {
          mymin = mymin_block[i];
        }
      }
      m_minloc = mymin.loc;

    }
    stopTimer();

    RAJAPERF_HIP_REDUCER_TEARDOWN(dminloc, mymin_block);

  } else {
     getCout() << "\n  FIRST_MIN : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename MappingHelper >
void FIRST_MIN::runHipVariantRAJA(VariantID vid)
{
  using reduction_policy = RAJA::hip_reduce;

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  FIRST_MIN_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceMinLoc<reduction_policy, Real_type, Index_type> loc(
                                                        m_xmin_init, m_initloc);

       RAJA::forall<exec_policy>( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIRST_MIN_BODY_RAJA;
       });

       m_minloc = loc.getLoc();

    }
    stopTimer();

  } else {
     getCout() << "\n  FIRST_MIN : Unknown Hip variant id = " << vid << std::endl;
  }
}

void FIRST_MIN::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_HIP ) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runHipVariantBase<decltype(block_size){},
                                decltype(mapping_helper)>(vid);

            }

            t += 1;

          } else if ( vid == RAJA_HIP ) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runHipVariantRAJA<decltype(block_size){},
                                decltype(mapping_helper)>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  FIRST_MIN : Unknown Hip variant id = " << vid << std::endl;

  }

}

void FIRST_MIN::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_HIP ) {

            auto algorithm_helper = gpu_algorithm::block_host_helper{};

            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      std::to_string(block_size));

          } else if ( vid == RAJA_HIP ) {

            auto algorithm_helper = gpu_algorithm::block_device_helper{};

            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      std::to_string(block_size));

          }

        });

      }

    });

  }

}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
