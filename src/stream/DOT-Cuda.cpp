//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace rajaperf
{
namespace stream
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void dot(Real_ptr a, Real_ptr b,
                    Real_ptr dprod, Real_type dprod_init,
                    Index_type iend)
{
  extern __shared__ Real_type pdot[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  pdot[ threadIdx.x ] = dprod_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    pdot[ threadIdx.x ] += a[ i ] * b[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      pdot[ threadIdx.x ] += pdot[ threadIdx.x + i ];
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( dprod, pdot[ 0 ] );
  }
}


template < size_t block_size, typename MappingHelper >
void DOT::runCudaVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  DOT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, dprod, hdprod, 1, 1);

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, (dot<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_dot_init, dprod, hdprod, 1, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( (dot<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          a, b, dprod, m_dot_init, iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(dprod, hdprod, 1, 1);
      m_dot += hdprod[0];

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(dprod, hdprod);

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void DOT::runCudaVariantRAJA(VariantID vid)
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

  DOT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<reduction_policy, Real_type> dot(m_dot_init);

       RAJA::forall<exec_policy>( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size, typename MappingHelper >
void DOT::runCudaVariantRAJANewReduce(VariantID vid)
{
  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::cuda_exec<block_size, true /*async*/>,
      RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  DOT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       Real_type tdot = m_dot_init;

       RAJA::forall<exec_policy>( res,
         RAJA::RangeSegment(ibegin, iend),
         RAJA::expt::Reduce<RAJA::operators::plus>(&tdot),
         [=] __device__ (Index_type i, Real_type& dot) {
           DOT_BODY;
         }
       );

       m_dot += static_cast<Real_type>(tdot);

    }
    stopTimer();

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void DOT::runCudaVariant(VariantID vid, size_t tune_idx)
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

    getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void DOT::setCudaTuningDefinitions(VariantID vid)
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

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
