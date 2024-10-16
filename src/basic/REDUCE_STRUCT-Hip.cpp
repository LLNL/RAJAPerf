//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

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
__global__ void reduce_struct(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax, 
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax, 
                              Real_type m_init_sum,
                              Real_type m_init_min,
                              Real_type m_init_max,
                              Index_type iend)
{

  //x
  HIP_DYNAMIC_SHARED( Real_type, shared)
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
    RAJA::atomicAdd<RAJA::hip_atomic>( xsum, pxsum[ 0 ] );
    RAJA::atomicMin<RAJA::hip_atomic>( xmin, pxmin[ 0 ] );
    RAJA::atomicMax<RAJA::hip_atomic>( xmax, pxmax[ 0 ] );

    RAJA::atomicAdd<RAJA::hip_atomic>( ysum, pysum[ 0 ] );
    RAJA::atomicMin<RAJA::hip_atomic>( ymin, pymin[ 0 ] );
    RAJA::atomicMax<RAJA::hip_atomic>( ymax, pymax[ 0 ] );
  }
}

template < size_t block_size, typename MappingHelper >
void REDUCE_STRUCT::runHipVariantBase(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    RAJAPERF_HIP_REDUCER_SETUP(Real_ptr, mem, hmem, 6, 1);

    constexpr size_t shmem = 6*sizeof(Real_type)*block_size;
    const size_t max_grid_size = RAJAPERF_HIP_GET_MAX_BLOCKS(
        MappingHelper, (reduce_struct<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type imem[6] {m_init_sum, m_init_min, m_init_max, m_init_sum, m_init_min, m_init_max};
      RAJAPERF_HIP_REDUCER_INITIALIZE(imem, mem, hmem, 6, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchHipKernel( (reduce_struct<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         points.x, points.y,
                         mem, mem+1, mem+2,    // xcenter,xmin,xmax
                         mem+3, mem+4, mem+5,  // ycenter,ymin,ymax
                         m_init_sum, m_init_min, m_init_max,
                         points.N ); 

      RAJAPERF_HIP_REDUCER_COPY_BACK(mem, hmem, 6, 1);
      points.SetCenter(hmem[0]/points.N, hmem[3]/points.N);
      points.SetXMin(hmem[1]);
      points.SetXMax(hmem[2]);
      points.SetYMin(hmem[4]);
      points.SetYMax(hmem[5]);
      m_points = points;

    }
    stopTimer();

    RAJAPERF_HIP_REDUCER_TEARDOWN(mem, hmem);

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;
  }

}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE_STRUCT::runHipVariantRAJA(VariantID vid)
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

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

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
      m_points = points;

    }
    stopTimer();

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;
  }

}

template < size_t block_size, typename MappingHelper >
void REDUCE_STRUCT::runHipVariantRAJANewReduce(VariantID vid)
{
  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type txsum = m_init_sum;
      Real_type tysum = m_init_sum;
      Real_type txmin = m_init_min;
      Real_type tymin = m_init_min;
      Real_type txmax = m_init_max;
      Real_type tymax = m_init_max;

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&txsum),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tysum),
        RAJA::expt::Reduce<RAJA::operators::minimum>(&txmin),
        RAJA::expt::Reduce<RAJA::operators::minimum>(&tymin),
        RAJA::expt::Reduce<RAJA::operators::maximum>(&txmax),
        RAJA::expt::Reduce<RAJA::operators::maximum>(&tymax),
        [=] __device__ (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& xsum,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& ysum,
          RAJA::expt::ValOp<Real_type, RAJA::operators::minimum>& xmin,
          RAJA::expt::ValOp<Real_type, RAJA::operators::minimum>& ymin,
          RAJA::expt::ValOp<Real_type, RAJA::operators::maximum>& xmax,
          RAJA::expt::ValOp<Real_type, RAJA::operators::maximum>& ymax ) {
          REDUCE_STRUCT_BODY_RAJA;
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
     getCout() << "\n  REDUCE_STRUCT : Unknown HIP variant id = " << vid << std::endl;
  }

}

void REDUCE_STRUCT::runHipVariant(VariantID vid, size_t tune_idx)
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

    getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;

  }

}

void REDUCE_STRUCT::setHipTuningDefinitions(VariantID vid)
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
