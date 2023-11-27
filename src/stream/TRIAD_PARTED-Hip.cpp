//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

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
void TRIAD_PARTED::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  TRIAD_PARTED_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((triad_parted<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  a, b, c, alpha,
                                          ibegin, iend );
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_lambda = [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        };

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_parted_lambda)>),
          grid_size, block_size, shmem, res.get_stream(), ibegin, iend, triad_parted_lambda);
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED::runHipVariantStream(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Hip> res;
  res.reserve(parts.size());
  res.emplace_back(getHipResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((triad_parted<block_size>), dim3(grid_size), dim3(block_size), shmem, res[p].get_stream(),  a, b, c, alpha,
                                          ibegin, iend );
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_lambda = [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        };

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_parted_lambda)>),
          grid_size, block_size, shmem, res[p].get_stream(), ibegin, iend, triad_parted_lambda);
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRIAD_PARTED::runHipVariantStreamOpenmp(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Hip> res;
  res.reserve(parts.size());
  res.emplace_back(getHipResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((triad_parted<block_size>), dim3(grid_size), dim3(block_size), shmem, res[p].get_stream(),  a, b, c, alpha,
                                          ibegin, iend );
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_lambda = [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        };

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_parted_lambda)>),
          grid_size, block_size, shmem, res[p].get_stream(), ibegin, iend, triad_parted_lambda);
        hipErrchk( hipGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Hip variant id = " << vid << std::endl;
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}



template < size_t block_size >
void TRIAD_PARTED::runHipVariantStreamEvent(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Hip> res;
  res.reserve(parts.size());
  res.emplace_back(getHipResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  std::vector<hipEvent_t> events(parts.size(), hipEvent_t{});
  for (size_t p = 0; p < parts.size(); ++p ) {
    hipErrchk( hipEventCreateWithFlags( &events[p], hipEventDisableTiming ) );
  }

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipEventRecord( events[0], res[0].get_stream() ) );

      for (size_t p = 1; p < parts.size(); ++p ) {
        hipErrchk( hipStreamWaitEvent( res[p].get_stream(), events[0], 0 ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((triad_parted<block_size>), dim3(grid_size), dim3(block_size), shmem, res[p].get_stream(),  a, b, c, alpha,
                                          ibegin, iend );
        hipErrchk( hipGetLastError() );

        hipErrchk( hipEventRecord( events[p], res[p].get_stream() ) );
        hipErrchk( hipStreamWaitEvent( res[0].get_stream(), events[p], 0 ) );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipEventRecord( events[0], res[0].get_stream() ) );

      for (size_t p = 1; p < parts.size(); ++p ) {
        hipErrchk( hipStreamWaitEvent( res[p].get_stream(), events[0], 0 ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_lambda = [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        };

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_parted_lambda)>),
          grid_size, block_size, shmem, res[p].get_stream(), ibegin, iend, triad_parted_lambda);
        hipErrchk( hipGetLastError() );

        hipErrchk( hipEventRecord( events[p], res[p].get_stream() ) );
        hipErrchk( hipStreamWaitEvent( res[0].get_stream(), events[p], 0 ) );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      camp::resources::Event e0 = res[0].get_event_erased();

      for (size_t p = 1; p < parts.size(); ++p ) {
        res[p].wait_for(&e0);

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        camp::resources::Event ep = RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });

        res[0].wait_for(&ep);
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Hip variant id = " << vid << std::endl;
  }

  for (size_t p = 0; p < parts.size(); ++p ) {
    hipErrchk( hipEventDestroy( events[p] ) );
  }
}

template < size_t block_size >
void TRIAD_PARTED::runHipVariantStreamEventOpenmp(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  const Index_type run_reps = getRunReps();

  TRIAD_PARTED_DATA_SETUP;

  std::vector<camp::resources::Hip> res;
  res.reserve(parts.size());
  res.emplace_back(getHipResource());
  for (size_t p = 1; p < parts.size(); ++p ) {
    res.emplace_back(p-1);
  }

  std::vector<hipEvent_t> events(parts.size(), hipEvent_t{});
  for (size_t p = 0; p < parts.size(); ++p ) {
    hipErrchk( hipEventCreateWithFlags( &events[p], hipEventDisableTiming ) );
  }

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipEventRecord( events[0], res[0].get_stream() ) );

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        hipErrchk( hipStreamWaitEvent( res[p].get_stream(), events[0], 0 ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((triad_parted<block_size>), dim3(grid_size), dim3(block_size), shmem, res[p].get_stream(),  a, b, c, alpha,
                                          ibegin, iend );
        hipErrchk( hipGetLastError() );

        hipErrchk( hipEventRecord( events[p], res[p].get_stream() ) );
        hipErrchk( hipStreamWaitEvent( res[0].get_stream(), events[p], 0 ) );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipEventRecord( events[0], res[0].get_stream() ) );

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        hipErrchk( hipStreamWaitEvent( res[p].get_stream(), events[0], 0 ) );

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        auto triad_parted_lambda = [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        };

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_parted_lambda)>),
          grid_size, block_size, shmem, res[p].get_stream(), ibegin, iend, triad_parted_lambda);
        hipErrchk( hipGetLastError() );

        hipErrchk( hipEventRecord( events[p], res[p].get_stream() ) );
        hipErrchk( hipStreamWaitEvent( res[0].get_stream(), events[p], 0 ) );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      camp::resources::Event e0 = res[0].get_event_erased();

      #pragma omp parallel for default(shared)
      for (size_t p = 1; p < parts.size(); ++p ) {
        res[p].wait_for(&e0);

        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        camp::resources::Event ep = RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res[p],
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });

        res[0].wait_for(&ep);
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Hip variant id = " << vid << std::endl;
  }

  for (size_t p = 0; p < parts.size(); ++p ) {
    hipErrchk( hipEventDestroy( events[p] ) );
  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

void TRIAD_PARTED::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

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
        runHipVariantStream<block_size>(vid);

      }

      t += 1;

      if (tune_idx == t) {

        setBlockSize(block_size);
        runHipVariantStreamOpenmp<block_size>(vid);

      }

      t += 1;

      if (tune_idx == t) {

        setBlockSize(block_size);
        runHipVariantStreamEvent<block_size>(vid);

      }

      t += 1;

      if (tune_idx == t) {

        setBlockSize(block_size);
        runHipVariantStreamEventOpenmp<block_size>(vid);

      }

      t += 1;

    }

  });
}

void TRIAD_PARTED::setHipTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

      addVariantTuningName(vid, "stream_"+std::to_string(block_size));

      addVariantTuningName(vid, "stream_omp_"+std::to_string(block_size));

      addVariantTuningName(vid, "stream_event_"+std::to_string(block_size));

      addVariantTuningName(vid, "stream_event_omp_"+std::to_string(block_size));

    }

  });
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
