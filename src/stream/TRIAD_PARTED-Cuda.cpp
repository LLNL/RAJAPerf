//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted(Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha,
                      Index_type ibegin, Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x + ibegin;
  if (i < iend) {
    TRIAD_PARTED_BODY;
  }
}


template < size_t block_size >
void TRIAD_PARTED::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  TRIAD_PARTED_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED::runCudaVariantStream(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Cuda> res;
  res.reserve(parts.size());
  res.emplace_back(getCudaResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED::runCudaVariantStreamOpenmp(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Cuda> res;
  res.reserve(parts.size());
  res.emplace_back(getCudaResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

template < size_t block_size >
void TRIAD_PARTED::runCudaVariantStreamEvent(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Cuda> res;
  res.reserve(parts.size());
  res.emplace_back(getCudaResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  std::vector<cudaEvent_t> events(parts.size(), cudaEvent_t{});
  for (size_t p = 0; p < parts.size(); ++p ) {
    cudaErrchk( cudaEventCreateWithFlags( &events[p], cudaEventDisableTiming ) );
  }

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaEventRecord( events[0], res[0].get_stream() ) );

      for (size_t p = 1; p < parts.size(); ++p ) {
        cudaErrchk( cudaStreamWaitEvent( res[p].get_stream(), events[0] ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );

        cudaErrchk( cudaEventRecord( events[p], res[p].get_stream() ) );
        cudaErrchk( cudaStreamWaitEvent( res[0].get_stream(), events[p] ) );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaEventRecord( events[0], res[0].get_stream() ) );

      for (size_t p = 1; p < parts.size(); ++p ) {
        cudaErrchk( cudaStreamWaitEvent( res[p].get_stream(), events[0] ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );

        cudaErrchk( cudaEventRecord( events[p], res[p].get_stream() ) );
        cudaErrchk( cudaStreamWaitEvent( res[0].get_stream(), events[p] ) );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      camp::resources::Event e0 = res[0].get_event_erased();

      for (size_t p = 1; p < parts.size(); ++p ) {
        res[p].wait_for(&e0);

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        camp::resources::Event ep = RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });

        res[0].wait_for(&ep);
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }

  for (size_t p = 0; p < parts.size(); ++p ) {
    cudaErrchk( cudaEventDestroy( events[p] ) );
  }
}

template < size_t block_size >
void TRIAD_PARTED::runCudaVariantStreamEventOpenmp(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Cuda> res;
  res.reserve(parts.size());
  res.emplace_back(getCudaResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  std::vector<cudaEvent_t> events(parts.size(), cudaEvent_t{});
  for (size_t p = 0; p < parts.size(); ++p ) {
    cudaErrchk( cudaEventCreateWithFlags( &events[p], cudaEventDisableTiming ) );
  }

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaEventRecord( events[0], res[0].get_stream() ) );

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        cudaErrchk( cudaStreamWaitEvent( res[p].get_stream(), events[0] ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );

        cudaErrchk( cudaEventRecord( events[p], res[p].get_stream() ) );
        cudaErrchk( cudaStreamWaitEvent( res[0].get_stream(), events[p] ) );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      cudaErrchk( cudaEventRecord( events[0], res[0].get_stream() ) );

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        cudaErrchk( cudaStreamWaitEvent( res[p].get_stream(), events[0] ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res[p].get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );

        cudaErrchk( cudaEventRecord( events[p], res[p].get_stream() ) );
        cudaErrchk( cudaStreamWaitEvent( res[0].get_stream(), events[p] ) );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      camp::resources::Event e0 = res[0].get_event_erased();

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        res[p].wait_for(&e0);

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        camp::resources::Event ep = RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });

        res[0].wait_for(&ep);
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }

  for (size_t p = 0; p < parts.size(); ++p ) {
    cudaErrchk( cudaEventDestroy( events[p] ) );
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void TRIAD_PARTED::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        setBlockSize(block_size);
        runCudaVariantBlock<block_size>(vid);

      }

      t += 1;

      if (tune_idx == t) {

        setBlockSize(block_size);
        runCudaVariantStream<block_size>(vid);

      }

      t += 1;

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      if (tune_idx == t) {

        setBlockSize(block_size);
        runCudaVariantStreamOpenmp<block_size>(vid);

      }

      t += 1;
#endif

      if (tune_idx == t) {

        setBlockSize(block_size);
        runCudaVariantStreamEvent<block_size>(vid);

      }

      t += 1;

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      if (tune_idx == t) {

        setBlockSize(block_size);
        runCudaVariantStreamEventOpenmp<block_size>(vid);

      }

      t += 1;
#endif

    }

  });
}

void TRIAD_PARTED::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

      addVariantTuningName(vid, "stream_"+std::to_string(block_size));

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      addVariantTuningName(vid, "stream_omp_"+std::to_string(block_size));
#endif

      addVariantTuningName(vid, "stream_event_"+std::to_string(block_size));

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      addVariantTuningName(vid, "stream_event_omp_"+std::to_string(block_size));
#endif

    }

  });
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
