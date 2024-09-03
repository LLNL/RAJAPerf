//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace rajaperf
{
namespace basic
{


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_type vsum_init,
                           Int_ptr vmin, Int_type vmin_init,
                           Int_ptr vmax, Int_type vmax_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED( Int_type, psum)
  Int_type* pmin = (Int_type*)&psum[ 1 * block_size ];
  Int_type* pmax = (Int_type*)&psum[ 2 * block_size ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psum[ threadIdx.x ] = vsum_init;
  pmin[ threadIdx.x ] = vmin_init;
  pmax[ threadIdx.x ] = vmax_init;

  for ( ; i < iend ; i += gridDim.x * block_size ) {
    psum[ threadIdx.x ] += vec[ i ];
    pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], vec[ i ] );
    pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], vec[ i ] );
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
      pmin[ threadIdx.x ] = RAJA_MIN( pmin[ threadIdx.x ], pmin[ threadIdx.x + i ] );
      pmax[ threadIdx.x ] = RAJA_MAX( pmax[ threadIdx.x ], pmax[ threadIdx.x + i ] );
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::hip_atomic>( vsum, psum[ 0 ] );
    RAJA::atomicMin<RAJA::hip_atomic>( vmin, pmin[ 0 ] );
    RAJA::atomicMax<RAJA::hip_atomic>( vmax, pmax[ 0 ] );
  }
}


template < size_t block_size, typename MappingHelper >
void REDUCE3_INT::runHipVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    RAJAPERF_HIP_REDUCER_SETUP(Int_ptr, vmem, hvmem, 3, 1);

    constexpr size_t shmem = 3*sizeof(Int_type)*block_size;
    const size_t max_grid_size = RAJAPERF_HIP_GET_MAX_BLOCKS(
        MappingHelper, (reduce3int<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Int_type ivmem[3] {m_vsum_init, m_vmin_init, m_vmax_init};
      RAJAPERF_HIP_REDUCER_INITIALIZE(ivmem, vmem, hvmem, 3, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchHipKernel( (reduce3int<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         vec,
                         vmem + 0, m_vsum_init,
                         vmem + 1, m_vmin_init,
                         vmem + 2, m_vmax_init,
                         iend );

      RAJAPERF_HIP_REDUCER_COPY_BACK(vmem, hvmem, 3, 1);
      m_vsum += hvmem[0];
      m_vmin = RAJA_MIN(m_vmin, hvmem[1]);
      m_vmax = RAJA_MAX(m_vmax, hvmem[2]);

    }
    stopTimer();

    RAJAPERF_HIP_REDUCER_TEARDOWN(vmem, hvmem);

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE3_INT::runHipVariantRAJA(VariantID vid)
{
  using reduction_policy = std::conditional_t<AlgorithmHelper::atomic,
      RAJA::hip_reduce_atomic,
      RAJA::hip_reduce>;

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<reduction_policy, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<reduction_policy, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<reduction_policy, Int_type> vmax(m_vmax_init);

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename MappingHelper >
void REDUCE3_INT::runHipVariantRAJANewReduce(VariantID vid)
{
  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Int_type tvsum = m_vsum_init;
      Int_type tvmin = m_vmin_init;
      Int_type tvmax = m_vmax_init;

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tvsum),
        RAJA::expt::Reduce<RAJA::operators::minimum>(&tvmin),
        RAJA::expt::Reduce<RAJA::operators::maximum>(&tvmax),
        [=] __device__ (Index_type i,
                        Int_type& vsum, Int_type& vmin, Int_type& vmax) {
          REDUCE3_INT_BODY;
        }
      );

      m_vsum += static_cast<Int_type>(tvsum);
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(tvmin));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(tvmax));

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

void REDUCE3_INT::runHipVariant(VariantID vid, size_t tune_idx)
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

            seq_for(gpu_algorithm::reducer_helpers{}, [&](auto algorithm_helper) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runHipVariantRAJA<decltype(block_size){},
                                  decltype(algorithm_helper),
                                  decltype(mapping_helper)>(vid);

              }

              t += 1;

            });

            if (tune_idx == t) {

              setBlockSize(block_size);
              runHipVariantRAJANewReduce<decltype(block_size){},
                                         decltype(mapping_helper)>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  REDUCE3_INT : Unknown Hip variant id = " << vid << std::endl;

  }

}

void REDUCE3_INT::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_HIP ) {

            auto algorithm_helper = gpu_algorithm::block_atomic_helper{};

            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      std::to_string(block_size));

          } else if ( vid == RAJA_HIP ) {

            seq_for(gpu_algorithm::reducer_helpers{}, [&](auto algorithm_helper) {

              addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                        decltype(mapping_helper)::get_name()+"_"+
                                        std::to_string(block_size));

            });

            auto algorithm_helper = gpu_algorithm::block_device_helper{};

            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      "new_"+std::to_string(block_size));

          }

        });

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
