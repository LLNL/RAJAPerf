//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "cub/device/device_histogram.cuh"
#include "cub/util_allocator.cuh"

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

// for these models the input is block_size and the output is values
using histogram_shared_atomic_model = ConstantModel<16>;

// for these models the input is block_size and the output is cache lines
using histogram_global_atomic_model = CutoffModel<512, 2, 1>;

// v100
// 10 bins - 1 bin per iterate - single bin
// shared  ConstantModel<16>        global  ConstantModel<1>
// 10 bins - 1 bin per iterate - random sized runs
// shared  ConstantModel<16>        global  CutoffModel<512, 2, 1>
// 10 bins - 1 bin per iterate - random bin
// shared  ConstantModel<8>         global  ConstantModel<1>
//
// 100 bins - 1 bin per iterate - single bin
// shared  ConstantModel<>        global  ConstantModel<>
// 100 bins - 1 bin per iterate - random sized runs
// shared  ConstantModel<>        global  ConstantModel<>
// 100 bins - 1 bin per iterate - random bin
// shared  ConstantModel<>         global  ConstantModel<>


template < size_t t_block_size, typename T, typename FunctionSignature >
struct histogram_info
{
  static constexpr size_t block_size = t_block_size;

  static size_t get_grid_size(size_t problem_size)
  {
    return RAJA_DIVIDE_CEILING_INT(problem_size, block_size);
  }

  static size_t get_max_shmem(FunctionSignature const& func)
  {
    cudaFuncAttributes func_attr;
    cudaErrchk(cudaFuncGetAttributes(&func_attr, (const void*)func));
    return func_attr.maxDynamicSharedSizeBytes;
  }

  FunctionSignature const& func;
  const size_t grid_size;
  const MultiReduceAtomicCalculator<T, Index_type> atomic_calc;

  template < typename GlobalModel, typename SharedModel >
  histogram_info(FunctionSignature const& a_func, size_t problem_size, size_t num_bins,
                 GlobalModel const& global_atomic_model, SharedModel const& shared_atomic_model)
    : func(a_func)
    , grid_size(get_grid_size(problem_size))
    , atomic_calc(num_bins, block_size, grid_size, get_max_shmem(a_func),
                  global_atomic_model, shared_atomic_model)
  { }

  std::string get_name() const
  {
    return "atomic_shared("+std::to_string(atomic_calc.shared_replication())+
           ")_global("+std::to_string(atomic_calc.global_replication())+
           ")_"+std::to_string(block_size);
  }
};


template < Index_type block_size, Index_type global_replication >
__launch_bounds__(block_size)
__global__ void histogram_atomic_global(HISTOGRAM::Data_ptr counts,
                                        Index_ptr bins,
                                        Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], i, global_replication), HISTOGRAM::Data_type(1));
  }
}

template < Index_type block_size, Index_type shared_replication, Index_type global_replication >
__launch_bounds__(block_size)
__global__ void histogram_atomic_shared_global(HISTOGRAM::Data_ptr global_counts,
                                        Index_ptr bins,
                                        Index_type num_bins,
                                        Index_type iend)
{
  extern __shared__ HISTOGRAM::Data_type shared_counts[];
  for (Index_type i = threadIdx.x; i < num_bins * shared_replication; i += block_size) {
    shared_counts[i] = HISTOGRAM::Data_type(0);
  }
  __syncthreads();

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;
    if (i < iend) {
      HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, shared_counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], threadIdx.x, shared_replication), HISTOGRAM::Data_type(1));
    }
  }
  __syncthreads();

  for (Index_type b = threadIdx.x; b < num_bins; b += block_size) {
    Index_type i = blockIdx.x * num_bins + b;
    auto block_sum = HISTOGRAM::Data_type(0);
    for (Index_type s = 0; s < shared_replication; ++s) {
      block_sum += shared_counts[HISTOGRAM_GPU_BIN_INDEX(b, s, shared_replication)];
    }
    HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, global_counts, HISTOGRAM_GPU_BIN_INDEX(b, i, global_replication), block_sum);
  }
}

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void histogram_atomic_runtime(HISTOGRAM::Data_ptr global_counts,
                                        Index_ptr bins,
                                        Index_type iend,
                                        MultiReduceAtomicCalculator<HISTOGRAM::Data_type, Index_type> atomic_calc)
{
  extern __shared__ HISTOGRAM::Data_type shared_counts[];
  for (Index_type i = threadIdx.x;
       i < Index_type(atomic_calc.num_bins() * atomic_calc.shared_replication());
       i += block_size) {
    shared_counts[i] = HISTOGRAM::Data_type(0);
  }
  __syncthreads();

  {
    Index_type i = blockIdx.x * block_size + threadIdx.x;
    if (i < iend) {
      HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, shared_counts, atomic_calc.get_shared_offset(bins[i], threadIdx.x), HISTOGRAM::Data_type(1));
    }
  }
  __syncthreads();

  for (Index_type b = threadIdx.x; b < atomic_calc.num_bins(); b += block_size) {
    auto block_sum = HISTOGRAM::Data_type(0);
    for (Index_type s = 0; s < atomic_calc.shared_replication(); ++s) {
      block_sum += shared_counts[atomic_calc.get_shared_offset(b, s)];
    }
    if (block_sum != HISTOGRAM::Data_type(0)) {
      HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, global_counts, atomic_calc.get_global_offset(b, blockIdx.x), block_sum);
    }
  }
}


void HISTOGRAM::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, 1);

  RAJAPERF_UNUSED_VAR(counts_init);

  if ( vid == Base_CUDA ) {

    cudaStream_t stream = res.get_stream();

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                     temp_storage_bytes,
                                                     bins+ibegin,
                                                     counts,
                                                     static_cast<int>(num_bins+1),
                                                     static_cast<Index_type>(0),
                                                     num_bins,
                                                     len,
                                                     stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                       temp_storage_bytes,
                                                       bins+ibegin,
                                                       counts,
                                                       static_cast<int>(num_bins+1),
                                                       static_cast<Index_type>(0),
                                                       num_bins,
                                                       len,
                                                       stream));

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, 1);
      HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, 1);

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}

template < size_t block_size, size_t global_replication,
           bool warp_atomics, bool bunched_atomics >
void HISTOGRAM::runCudaVariantAtomicGlobal(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, global_replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (histogram_atomic_global<block_size, global_replication>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          counts,
                          bins,
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      auto histogram_lambda = [=] __device__ (Index_type i) {
        HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], i, global_replication), HISTOGRAM::Data_type(1));
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(histogram_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, histogram_lambda );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using multi_reduce_policy = RAJA::policy::cuda::cuda_multi_reduce_policy<
        RAJA::cuda::MultiReduceTuning<
          RAJA::cuda::multi_reduce_algorithm::init_host_combine_global_atomic,
          void,
          RAJA::cuda::AtomicReplicationTuning<
            RAJA::cuda::GlobalAtomicReplicationMinPow2Concretizer<
              RAJA::cuda::ConstantPreferredReplicationConcretizer<global_replication>>,
            std::conditional_t<warp_atomics, RAJA::cuda::warp_global_xyz<>, RAJA::cuda::block_xyz<>>,
            std::conditional_t<bunched_atomics, RAJA::GetOffsetLeftBunched<0,int>, RAJA::GetOffsetLeft<int>>>>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      HISTOGRAM_INIT_COUNTS_RAJA(multi_reduce_policy);

      RAJA::forall<RAJA::cuda_exec<block_size, true /*async*/>>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=] __device__ (Index_type i) {
        HISTOGRAM_BODY;
      });

      HISTOGRAM_FINALIZE_COUNTS_RAJA(multi_reduce_policy);

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}

template < size_t block_size, size_t shared_replication, size_t global_replication,
           bool warp_atomics, bool bunched_atomics >
void HISTOGRAM::runCudaVariantAtomicShared(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, global_replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t shmem = num_bins*shared_replication*sizeof(Data_type);

      RPlaunchCudaKernel( (histogram_atomic_shared_global<block_size, shared_replication, global_replication>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          counts,
                          bins,
                          num_bins,
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using multi_reduce_policy = RAJA::policy::cuda::cuda_multi_reduce_policy<
        RAJA::cuda::MultiReduceTuning<
          RAJA::cuda::multi_reduce_algorithm::init_host_combine_block_then_grid_atomic,
          RAJA::cuda::AtomicReplicationTuning<
            RAJA::cuda::SharedAtomicReplicationMaxPow2Concretizer<
              RAJA::cuda::ConstantPreferredReplicationConcretizer<shared_replication>>,
            RAJA::cuda::thread_xyz<>,
            RAJA::GetOffsetRight<int>>,
          RAJA::cuda::AtomicReplicationTuning<
            RAJA::cuda::GlobalAtomicReplicationMinPow2Concretizer<
              RAJA::cuda::ConstantPreferredReplicationConcretizer<global_replication>>,
            std::conditional_t<warp_atomics, RAJA::cuda::warp_global_xyz<>, RAJA::cuda::block_xyz<>>,
            std::conditional_t<bunched_atomics, RAJA::GetOffsetLeftBunched<0,int>, RAJA::GetOffsetLeft<int>>>>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      HISTOGRAM_INIT_COUNTS_RAJA(multi_reduce_policy);

      RAJA::forall<RAJA::cuda_exec<block_size, true /*async*/>>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=] __device__ (Index_type i) {
        HISTOGRAM_BODY;
      });

      HISTOGRAM_FINALIZE_COUNTS_RAJA(multi_reduce_policy);

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}


template < typename MultiReduceInfo >
void HISTOGRAM::runCudaVariantAtomicRuntime(MultiReduceInfo info, VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_DATA_SETUP;

  static constexpr size_t block_size = info.block_size;
  const size_t grid_size = info.grid_size;
  const auto atomic_calc = info.atomic_calc;
  const size_t global_replication = atomic_calc.global_replication();
  const size_t shmem = atomic_calc.shared_memory_in_bytes();

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, global_replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      RPlaunchCudaKernel( info.func,
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          counts,
                          bins,
                          iend,
                          atomic_calc );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      for (Index_type b = 0; b < num_bins; ++b) {
        Data_type count_final = 0;
        for (size_t r = 0; r < global_replication; ++r) {
          count_final += hcounts[atomic_calc.get_global_offset(b, r)];
        }
        counts_final[b] = count_final;
      }

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}


void HISTOGRAM::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantLibrary(vid);

    }

    t += 1;

  }

  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantAtomicGlobal<decltype(block_size)::value, global_replication, false, false>(vid);

            }

            t += 1;

            if ( vid == RAJA_CUDA ) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantAtomicGlobal<decltype(block_size)::value, global_replication, true, false>(vid);

              }

              t += 1;

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantAtomicGlobal<decltype(block_size)::value, global_replication, false, true>(vid);

              }

              t += 1;

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantAtomicGlobal<decltype(block_size)::value, global_replication, true, true>(vid);

              }

              t += 1;

            }

            seq_for(gpu_atomic_shared_replications_type{}, [&](auto shared_replication) {

              if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  runCudaVariantAtomicShared<decltype(block_size)::value,
                                             shared_replication,
                                             decltype(global_replication)::value, false, false>(vid);

                }

                t += 1;

              }

              if ( vid == RAJA_CUDA ) {

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  runCudaVariantAtomicShared<decltype(block_size)::value,
                                             shared_replication,
                                             decltype(global_replication)::value, true, false>(vid);

                }

                t += 1;

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  runCudaVariantAtomicShared<decltype(block_size)::value,
                                             shared_replication,
                                             decltype(global_replication)::value, false, true>(vid);

                }

                t += 1;

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  runCudaVariantAtomicShared<decltype(block_size)::value,
                                             shared_replication,
                                             decltype(global_replication)::value, true, true>(vid);

                }

                t += 1;

              }

              if ( vid == Base_CUDA ) {

                if (tune_idx == t) {

                  histogram_info<decltype(block_size)::value, Data_type, decltype(histogram_atomic_runtime<decltype(block_size)::value>)> info(
                      histogram_atomic_runtime<decltype(block_size)::value>, getActualProblemSize(), m_num_bins,
                      ConstantModel<decltype(global_replication)::value>{}, ConstantModel<shared_replication>{});
                  setBlockSize(block_size);
                  runCudaVariantAtomicRuntime(info, vid);

                }

                t += 1;

              }

            });

          }

        });

        if ( vid == Base_CUDA ) {

          if (tune_idx == t) {

            histogram_info<block_size, Data_type, decltype(histogram_atomic_runtime<block_size>)> info(
                histogram_atomic_runtime<block_size>, getActualProblemSize(), m_num_bins,
                histogram_global_atomic_model{}, histogram_shared_atomic_model{});
            setBlockSize(block_size);
            runCudaVariantAtomicRuntime(info, vid);

          }

          t += 1;

        }

      }

    });

  } else {

    getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void HISTOGRAM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            addVariantTuningName(vid, "atomic_global<"+std::to_string(global_replication)+
                                      ">block_unbunched_"+std::to_string(block_size));

            if ( vid == RAJA_CUDA ) {
              addVariantTuningName(vid, "atomic_global<"+std::to_string(global_replication)+
                                        ">warp_unbunched_"+std::to_string(block_size));
              addVariantTuningName(vid, "atomic_global<"+std::to_string(global_replication)+
                                        ">block_bunched_"+std::to_string(block_size));
              addVariantTuningName(vid, "atomic_global<"+std::to_string(global_replication)+
                                        ">warp_bunched_"+std::to_string(block_size));
            }

            seq_for(gpu_atomic_shared_replications_type{}, [&](auto shared_replication) {

              if ( vid == Base_CUDA || vid == RAJA_CUDA ) {
                addVariantTuningName(vid, "atomic_shared<"+std::to_string(shared_replication)+
                                          ">_global<"+std::to_string(global_replication)+
                                          ">block_unbunched_"+std::to_string(block_size));
              }

              if ( vid == RAJA_CUDA ) {
                addVariantTuningName(vid, "atomic_shared<"+std::to_string(shared_replication)+
                                          ">_global<"+std::to_string(global_replication)+
                                          ">warp_unbunched_"+std::to_string(block_size));
                addVariantTuningName(vid, "atomic_shared<"+std::to_string(shared_replication)+
                                          ">_global<"+std::to_string(global_replication)+
                                          ">block_bunched_"+std::to_string(block_size));
                addVariantTuningName(vid, "atomic_shared<"+std::to_string(shared_replication)+
                                          ">_global<"+std::to_string(global_replication)+
                                          ">warp_bunched_"+std::to_string(block_size));
              }

              if ( vid == Base_CUDA ) {
                histogram_info<decltype(block_size)::value, Data_type, decltype(histogram_atomic_runtime<decltype(block_size)::value>)> info(
                    histogram_atomic_runtime<decltype(block_size)::value>, getActualProblemSize(), m_num_bins,
                    ConstantModel<decltype(global_replication)::value>{}, ConstantModel<shared_replication>{});
                auto name = info.get_name();
                addVariantTuningName(vid, name.c_str());
              }

            });

          }

        });

        if ( vid == Base_CUDA ) {
          histogram_info<block_size, Data_type, decltype(histogram_atomic_runtime<block_size>)> info(
              histogram_atomic_runtime<block_size>, getActualProblemSize(), m_num_bins,
              histogram_global_atomic_model{}, histogram_shared_atomic_model{});
          auto name = info.get_name();
          addVariantTuningName(vid, name.c_str());
        }

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
