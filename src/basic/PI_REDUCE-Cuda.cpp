//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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



template < size_t block_size >
void PI_REDUCE::runCudaVariantBlockAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, pi, hpi, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = sizeof(Real_type)*block_size;

      RPlaunchCudaKernel( (pi_reduce<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          dx,
                          pi, m_pi_init,
                          iend );
      cudaErrchk( cudaGetLastError() );

      Real_type rpi;
      RAJAPERF_CUDA_REDUCER_COPY_BACK(&rpi, pi, hpi, 1);
      m_pi = rpi * static_cast<Real_type>(4);

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(pi, hpi);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce_atomic, Real_type> pi(m_pi_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
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

template < size_t block_size >
void PI_REDUCE::runCudaVariantBlockAtomicOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, pi, hpi, 1);

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = detail::getCudaOccupancyMaxBlocks(
        (pi_reduce<block_size>), block_size, shmem);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_pi_init, pi, hpi, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( (pi_reduce<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          dx,
                          pi, m_pi_init,
                          iend );
      cudaErrchk( cudaGetLastError() );

      Real_type rpi;
      RAJAPERF_CUDA_REDUCER_COPY_BACK(&rpi, pi, hpi, 1);
      m_pi = rpi * static_cast<Real_type>(4);

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(pi, hpi);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce_atomic, Real_type> pi(m_pi_init);

      RAJA::forall< RAJA::cuda_exec_occ_calc<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = 4.0 * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}


template < size_t block_size >
void PI_REDUCE::runCudaVariantBlockDevice(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> pi(m_pi_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = 4.0 * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void PI_REDUCE::runCudaVariantBlockDeviceOccGS(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  PI_REDUCE_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> pi(m_pi_init);

      RAJA::forall< RAJA::cuda_exec_occ_calc<block_size, true /*async*/> >( res,
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

void PI_REDUCE::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          setBlockSize(block_size);
          runCudaVariantBlockAtomic<block_size>(vid);

        }

        t += 1;

        if (tune_idx == t) {

          setBlockSize(block_size);
          runCudaVariantBlockAtomicOccGS<block_size>(vid);

        }

        t += 1;

        if ( vid == RAJA_CUDA ) {

          if (tune_idx == t) {

            setBlockSize(block_size);
            runCudaVariantBlockDevice<block_size>(vid);

          }

          t += 1;

          if (tune_idx == t) {

            setBlockSize(block_size);
            runCudaVariantBlockDeviceOccGS<block_size>(vid);

          }

          t += 1;

        }

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

        addVariantTuningName(vid, "blkatm_"+std::to_string(block_size));

        addVariantTuningName(vid, "blkatm_occgs_"+std::to_string(block_size));

        if ( vid == RAJA_CUDA ) {

          addVariantTuningName(vid, "blkdev_"+std::to_string(block_size));

          addVariantTuningName(vid, "blkdev_occgs_"+std::to_string(block_size));

        }

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
