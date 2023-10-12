//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::hip_atomic>( vsum, psum[ 0 ] );
    RAJA::atomicMin<RAJA::hip_atomic>( vmin, pmin[ 0 ] );
    RAJA::atomicMax<RAJA::hip_atomic>( vmax, pmax[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *vsum += psum[ 0 ];
    *vmin = RAJA_MIN( *vmin, pmin[ 0 ] );
    *vmax = RAJA_MAX( *vmax, pmax[ 0 ] );
  }
#endif
}



template < size_t block_size >
void REDUCE3_INT::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    Int_ptr vmem_init;
    allocData(DataSpace::HipPinned, vmem_init, 3);

    Int_ptr vmem;
    allocData(DataSpace::HipDevice, vmem, 3);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      hipErrchk( hipMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                 hipMemcpyHostToDevice, res.get_stream() ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 3*sizeof(Int_type)*block_size;
      hipLaunchKernelGGL((reduce3int<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
                                                    vec,
                                                    vmem + 0, m_vsum_init,
                                                    vmem + 1, m_vmin_init,
                                                    vmem + 2, m_vmax_init,
                                                    iend );
      hipErrchk( hipGetLastError() );

      Int_type lmem[3];
      hipErrchk( hipMemcpyAsync( &lmem[0], vmem, 3*sizeof(Int_type),
                                 hipMemcpyDeviceToHost, res.get_stream() ) );
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, vmem);
    deallocData(DataSpace::HipPinned, vmem_init);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::hip_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::hip_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
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

template < size_t block_size >
void REDUCE3_INT::runHipVariantOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    Int_ptr vmem_init;
    allocData(DataSpace::HipPinned, vmem_init, 3);

    Int_ptr vmem;
    allocData(DataSpace::HipDevice, vmem, 3);

    constexpr size_t shmem = 3*sizeof(Int_type)*block_size;
    const size_t max_grid_size = detail::getHipOccupancyMaxBlocks(
        (reduce3int<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = m_vsum_init;
      vmem_init[1] = m_vmin_init;
      vmem_init[2] = m_vmax_init;
      hipErrchk( hipMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                 hipMemcpyHostToDevice, res.get_stream() ) );

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);
      hipLaunchKernelGGL((reduce3int<block_size>), dim3(grid_size), dim3(block_size),
                                                   shmem, res.get_stream(),
                                                    vec,
                                                    vmem + 0, m_vsum_init,
                                                    vmem + 1, m_vmin_init,
                                                    vmem + 2, m_vmax_init,
                                                    iend );
      hipErrchk( hipGetLastError() );

      Int_type lmem[3];
      hipErrchk( hipMemcpyAsync( &lmem[0], vmem, 3*sizeof(Int_type),
                                 hipMemcpyDeviceToHost, res.get_stream() ) );
      hipErrchk( hipStreamSynchronize( res.get_stream() ) );
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    deallocData(DataSpace::HipDevice, vmem);
    deallocData(DataSpace::HipPinned, vmem_init);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::hip_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::hip_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::hip_exec_occ_calc<block_size, true /*async*/> >( res,
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

void REDUCE3_INT::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantBlock<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runHipVariantOccGS<block_size>(vid);

        }

        t += 1;

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

        addVariantTuningName(vid, "block_"+std::to_string(block_size));

        addVariantTuningName(vid, "occgs_"+std::to_string(block_size));

      }

    });

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
