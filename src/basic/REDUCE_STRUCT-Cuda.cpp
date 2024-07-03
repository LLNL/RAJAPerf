//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

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
__global__ void reduce_struct(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax, 
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax, 
                              Real_type m_init_sum, 
                              Real_type m_init_min, 
                              Real_type m_init_max, 
                              Index_type iend)
{

  //x
  extern __shared__ Real_type shared[];
  Real_type* pxsum = (Real_type*)&shared[ 0 * blockDim.x ];
  Real_type* pxmin = (Real_type*)&shared[ 1 * blockDim.x ];
  Real_type* pxmax = (Real_type*)&shared[ 2 * blockDim.x ];
  //y
  Real_type* pysum = (Real_type*)&shared[ 3 * blockDim.x ];
  Real_type* pymin = (Real_type*)&shared[ 4 * blockDim.x ];
  Real_type* pymax = (Real_type*)&shared[ 5 * blockDim.x ];

  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;

  //x
  pxsum[ threadIdx.x ] = m_init_sum;
  pxmin[ threadIdx.x ] = m_init_min;
  pxmax[ threadIdx.x ] = m_init_max;
  //y
  pysum[ threadIdx.x ] = m_init_sum;
  pymin[ threadIdx.x ] = m_init_min;
  pymax[ threadIdx.x ] = m_init_max;


  for ( ; i < iend ; i += gridDim.x * blockDim.x ) {
	//x
    pxsum[ threadIdx.x ] += x[ i ];
    pxmin[ threadIdx.x ] = RAJA_MIN( pxmin[ threadIdx.x ], x[ i ] );
    pxmax[ threadIdx.x ] = RAJA_MAX( pxmax[ threadIdx.x ], x[ i ] );
	//y
    pysum[ threadIdx.x ] += y[ i ];
    pymin[ threadIdx.x ] = RAJA_MIN( pymin[ threadIdx.x ], y[ i ] );
    pymax[ threadIdx.x ] = RAJA_MAX( pymax[ threadIdx.x ], y[ i ] );

  }
  __syncthreads();

  for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
	  //x
      pxsum[ threadIdx.x ] += pxsum[ threadIdx.x + i ];
      pxmin[ threadIdx.x ] = RAJA_MIN( pxmin[ threadIdx.x ], pxmin[ threadIdx.x + i ] );
      pxmax[ threadIdx.x ] = RAJA_MAX( pxmax[ threadIdx.x ], pxmax[ threadIdx.x + i ] );
	  //y
      pysum[ threadIdx.x ] += pysum[ threadIdx.x + i ];
      pymin[ threadIdx.x ] = RAJA_MIN( pymin[ threadIdx.x ], pymin[ threadIdx.x + i ] );
      pymax[ threadIdx.x ] = RAJA_MAX( pymax[ threadIdx.x ], pymax[ threadIdx.x + i ] );

    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( xsum, pxsum[ 0 ] );
    RAJA::atomicMin<RAJA::cuda_atomic>( xmin, pxmin[ 0 ] );
    RAJA::atomicMax<RAJA::cuda_atomic>( xmax, pxmax[ 0 ] );

    RAJA::atomicAdd<RAJA::cuda_atomic>( xsum, pysum[ 0 ] );
    RAJA::atomicMin<RAJA::cuda_atomic>( ymin, pymin[ 0 ] );
    RAJA::atomicMax<RAJA::cuda_atomic>( ymax, pymax[ 0 ] );
  }
}

template < size_t block_size, typename MappingHelper >
void REDUCE_STRUCT::runCudaVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, mem, hmem, 6);

    constexpr size_t shmem = 6*sizeof(Real_type)*block_size;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, (reduce_struct<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type imem[6] {m_init_sum, m_init_min, m_init_max, m_init_sum, m_init_min, m_init_max};
      RAJAPERF_CUDA_REDUCER_INITIALIZE(imem, mem, hmem, 6);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( (reduce_struct<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          points.x, points.y,
                          mem, mem+1, mem+2,    // xcenter,xmin,xmax
                          mem+3, mem+4, mem+5,  // ycenter,ymin,ymax
                          m_init_sum, m_init_min, m_init_max,
                          points.N );

      Real_type rmem[6];
      RAJAPERF_CUDA_REDUCER_COPY_BACK(rmem, mem, hmem, 6);
      points.SetCenter(rmem[0]/points.N, rmem[3]/points.N);
      points.SetXMin(rmem[1]);
      points.SetXMax(rmem[2]);
      points.SetYMin(rmem[4]);
      points.SetYMax(rmem[5]);
      m_points=points;

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(mem, hmem);

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown CUDA variant id = " << vid << std::endl;
  }

}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE_STRUCT::runCudaVariantRAJA(VariantID vid)
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

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<reduction_policy, Real_type> xsum(m_init_sum);
      RAJA::ReduceSum<reduction_policy, Real_type> ysum(m_init_sum);
      RAJA::ReduceMin<reduction_policy, Real_type> xmin(m_init_min);
      RAJA::ReduceMin<reduction_policy, Real_type> ymin(m_init_min);
      RAJA::ReduceMax<reduction_policy, Real_type> xmax(m_init_max);
      RAJA::ReduceMax<reduction_policy, Real_type> ymax(m_init_max);

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_STRUCT_BODY_RAJA;
      });

      points.SetCenter((xsum.get()/(points.N)),
                       (ysum.get()/(points.N)));
      points.SetXMin((xmin.get()));
      points.SetXMax((xmax.get()));
      points.SetYMin((ymin.get()));
      points.SetYMax((ymax.get()));
      m_points=points;

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown CUDA variant id = " << vid << std::endl;
  }

}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE_STRUCT::runCudaVariantRAJANewReduce(VariantID vid)
{
  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::cuda_exec<block_size, true /*async*/>,
      RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type txsum = m_init_sum;
      Real_type tysum = m_init_sum;
      Real_type txmin = m_init_min;
      Real_type tymin = m_init_min;
      Real_type txmax = m_init_max;
      Real_type tymax = m_init_max;

      RAJA::forall<exec_policy>(
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&txsum),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tysum),
        RAJA::expt::Reduce<RAJA::operators::minimum>(&txmin),
        RAJA::expt::Reduce<RAJA::operators::minimum>(&tymin),
        RAJA::expt::Reduce<RAJA::operators::maximum>(&txmax),
        RAJA::expt::Reduce<RAJA::operators::maximum>(&tymax),
        [=] __device__ (Index_type i, Real_type& xsum, Real_type& ysum,
                                      Real_type& xmin, Real_type& ymin,
                                      Real_type& xmax, Real_type& ymax) {
          REDUCE_STRUCT_BODY;
        }
      );

      points.SetCenter(static_cast<Real_type>(txsum)/(points.N),
                       static_cast<Real_type>(tysum)/(points.N));
      points.SetXMin(static_cast<Real_type>(txmin));
      points.SetXMax(static_cast<Real_type>(txmax));
      points.SetYMin(static_cast<Real_type>(tymin));
      points.SetYMax(static_cast<Real_type>(tymax));
      m_points = points;

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown CUDA variant id = " << vid << std::endl;
  }

}

void REDUCE_STRUCT::runCudaVariant(VariantID vid, size_t tune_idx)
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

              auto algorithm_helper = gpu_algorithm::block_device_helper{};
              setBlockSize(block_size);
              runCudaVariantRAJANewReduce<decltype(block_size){},
                                          decltype(algorithm_helper),
                                          decltype(mapping_helper)>(vid);
              RAJA_UNUSED_VAR(algorithm_helper); // to quiet compiler warning

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  REDUCE_STRUCT : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE_STRUCT::setCudaTuningDefinitions(VariantID vid)
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
