//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

namespace rajaperf
{
namespace basic
{

  
#define REDUCE_STRUCT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(points.x, m_x, points.N); \
  allocAndInitCudaDeviceData(points.y, m_y, points.N); \
  

#define REDUCE_STRUCT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(points.x); \
  deallocCudaDeviceData(points.y);

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

// serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( xsum, pxsum[ 0 ] );
    RAJA::atomicMin<RAJA::cuda_atomic>( xmin, pxmin[ 0 ] );
    RAJA::atomicMax<RAJA::cuda_atomic>( xmax, pxmax[ 0 ] );

    RAJA::atomicAdd<RAJA::cuda_atomic>( xsum, pysum[ 0 ] );
    RAJA::atomicMin<RAJA::cuda_atomic>( ymin, pymin[ 0 ] );
    RAJA::atomicMax<RAJA::cuda_atomic>( ymax, pymax[ 0 ] );
  }
}

template < size_t block_size >
void REDUCE_STRUCT::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE_STRUCT_DATA_SETUP_CUDA;

    Real_ptr mem; //xcenter,xmin,xmax,ycenter,ymin,ymax
    allocCudaDeviceData(mem,6);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk(cudaMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));  
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
                                                            
      reduce_struct<block_size><<<grid_size, block_size,
                                  6*sizeof(Real_type)*block_size>>>(
        points.x, points.y,
        mem, mem+1, mem+2,    // xcenter,xmin,xmax
        mem+3, mem+4, mem+5,  // ycenter,ymin,ymax
        m_init_sum, m_init_min, m_init_max,
        points.N);
      cudaErrchk( cudaGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getCudaDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/points.N, lmem[3]/points.N);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(mem);

  } else if ( vid == RAJA_CUDA ) {

    REDUCE_STRUCT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> xsum(m_init_sum);
      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> ysum(m_init_sum);
      RAJA::ReduceMin<RAJA::cuda_reduce, Real_type> xmin(m_init_min); 
      RAJA::ReduceMin<RAJA::cuda_reduce, Real_type> ymin(m_init_min);
      RAJA::ReduceMax<RAJA::cuda_reduce, Real_type> xmax(m_init_max); 
      RAJA::ReduceMax<RAJA::cuda_reduce, Real_type> ymax(m_init_max);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
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

    REDUCE_STRUCT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown CUDA variant id = " << vid << std::endl;
  }

}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(REDUCE_STRUCT, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
