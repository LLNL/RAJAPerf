//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

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
__global__ void pi_reduce(Real_type dx,
                          Real_ptr pi, Real_type pi_init,
                          Index_type iend)
{
  extern __shared__ Real_type ppi[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  ppi[ threadIdx.x ] = pi_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    double x = (double(i) + 0.5) * dx;
    ppi[ threadIdx.x ] += dx / (1.0 + x * x);
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      ppi[ threadIdx.x ] += ppi[ threadIdx.x + i ];
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( pi, ppi[ 0 ] );
  }
}


template < size_t block_size, typename MappingHelper >
void PI_REDUCE::runCudaVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, pi, hpi, 1, 1);

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, (pi_reduce<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( (pi_reduce<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          dx,
                          pi, m_pi_init,
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(pi, hpi, 1, 1);
      m_pi = hpi[0] * static_cast<Real_type>(4);

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(pi, hpi);

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void PI_REDUCE::runCudaVariantRAJA(VariantID vid)
{
  using reduction_policy = std::conditional_t<AlgorithmHelper::atomic,
      RAJA::cuda_reduce_atomic,
      RAJA::cuda_reduce>;

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::cuda_exec<block_size, true /*async*/>,
      RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<reduction_policy, Real_type> pi(m_pi_init);

      RAJA::forall<exec_policy>( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = static_cast<Real_type>(4) * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}


template < size_t block_size, typename MappingHelper >
void PI_REDUCE::runCudaVariantRAJANewReduce(VariantID vid)
{
  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::cuda_exec<block_size, true /*async*/>,
      RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type tpi = m_pi_init;

      RAJA::forall< exec_policy >( res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
        [=] __device__ (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& pi) {
          PI_REDUCE_BODY;
        }
      );

      m_pi = static_cast<Real_type>(tpi) * 4.0;

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void PI_REDUCE::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_CUDA ) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantBase<decltype(block_size){},
                                 decltype(mapping_helper)>(vid);

            }

            t += 1;

          } else if ( vid == RAJA_CUDA ) {

            seq_for(gpu_algorithm::reducer_helpers{}, [&](auto algorithm_helper) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantRAJA<decltype(block_size){},
                                   decltype(algorithm_helper),
                                   decltype(mapping_helper)>(vid);

              }

              t += 1;

            });

            if (tune_idx == t) {
  
              setBlockSize(block_size);
              runCudaVariantRAJANewReduce<decltype(block_size){},
                                          decltype(mapping_helper)>(vid);
  
            }
  
            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void PI_REDUCE::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_CUDA ) {

            auto algorithm_helper = gpu_algorithm::block_atomic_helper{};

            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      std::to_string(block_size));
            RAJA_UNUSED_VAR(algorithm_helper); // to quiet compiler warning

          } else if ( vid == RAJA_CUDA ) {

            seq_for(gpu_algorithm::reducer_helpers{}, [&](auto algorithm_helper) {

              addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                        decltype(mapping_helper)::get_name()+"_"+
                                        std::to_string(block_size));

            });
              
            auto algorithm_helper = gpu_algorithm::block_device_helper{};
              
            addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                      decltype(mapping_helper)::get_name()+"_"+
                                      "new_"+std::to_string(block_size));
            RAJA_UNUSED_VAR(algorithm_helper); // to quiet compiler warning

          }

        });

      }

    });

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
