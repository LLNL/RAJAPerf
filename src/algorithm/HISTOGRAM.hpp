//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HISTOGRAM kernel reference implementation:
///
/// Index_type* counts = calloc(num_bins, sizeof(Index_type));
/// for (Index_type i = 0; i < N; ++i ) {
///   counts[bins[i]] += 1;
/// }
///

#ifndef RAJAPerf_Algorithm_HISTOGRAM_HPP
#define RAJAPerf_Algorithm_HISTOGRAM_HPP

#define HISTOGRAM_DATA_SETUP \
  Index_type num_bins = m_num_bins; \
  Index_ptr bins = m_bins; \
  Data_ptr counts_init = m_counts_init.data(); \
  Data_ptr counts_final = m_counts_final.data(); \
  Data_ptr counts; \
  allocData(getReductionDataSpace(vid), counts, num_bins);

#define HISTOGRAM_DATA_TEARDOWN \
  deallocData(counts, vid);

#define HISTOGRAM_GPU_DATA_SETUP \
  Index_type num_bins = m_num_bins; \
  Index_ptr bins = m_bins; \
  Data_ptr counts_init = m_counts_init.data(); \
  Data_ptr counts_final = m_counts_final.data();

#define HISTOGRAM_BODY \
  counts[bins[i]] += static_cast<Data_type>(1);

#define HISTOGRAM_RAJA_BODY(policy) \
  RAJA::atomicAdd<policy>(&counts[bins[i]], static_cast<Data_type>(1));

#define HISTOGRAM_GPU_BIN_INDEX(bin, offset, replication) \
  ((bin)*(replication) + ((offset)%(replication)))

#define HISTOGRAM_GPU_RAJA_BODY(policy, counts, index, value) \
  RAJA::atomicAdd<policy>(&(counts)[(index)], (value));

#define HISTOGRAM_INIT_VALUES \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    counts[b] = counts_init[b]; \
  }

#define HISTOGRAM_FINALIZE_VALUES \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    counts_final[b] = counts[b]; \
  }

#define HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, replication) \
  for (Index_type b = 0; b < (num_bins); ++b) { \
    Data_type count_final = 0; \
    for (size_t r = 0; r < (replication); ++r) { \
      count_final += (hcounts)[HISTOGRAM_GPU_BIN_INDEX(b, r, replication)]; \
    } \
    counts_final[b] = count_final; \
  }


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class HISTOGRAM : public KernelBase
{
public:
  using Data_type = unsigned long long;
  using Data_ptr = Data_type*;

  HISTOGRAM(const RunParams& params);

  ~HISTOGRAM();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runKokkosVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void runCudaVariantLibrary(VariantID vid);
  void runHipVariantLibrary(VariantID vid);
  template < size_t block_size, size_t global_replication >
  void runCudaVariantAtomicGlobal(VariantID vid);
  template < size_t block_size, size_t global_replication >
  void runHipVariantAtomicGlobal(VariantID vid);
  template < size_t block_size, size_t shared_replication, size_t global_replication >
  void runCudaVariantAtomicShared(VariantID vid);
  template < size_t block_size, size_t shared_replication, size_t global_replication >
  void runHipVariantAtomicShared(VariantID vid);
  template < typename MultiReduceInfo >
  void runCudaVariantAtomicRuntime(MultiReduceInfo info, VariantID vid);
  template < typename MultiReduceInfo >
  void runHipVariantAtomicRuntime(MultiReduceInfo info, VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;
  static const size_t default_gpu_atomic_global_replication = 2048; // 512, 512
  // using gpu_atomic_global_replications_type = integer::make_atomic_replication_list_type<default_gpu_atomic_global_replication>;
  using gpu_atomic_global_replications_type = integer::list_type<32, 64, 128, 256, 512, 1024, 2048, 4096>;
  using gpu_atomic_shared_replications_type = integer::list_type<1, 2, 4, 8, 16, 32, 64>;

  Index_type m_num_bins;
  Index_ptr m_bins;
  std::vector<Data_type> m_counts_init;
  std::vector<Data_type> m_counts_final;
};


// Compute lhs % rhs between non-negative lhs and positive power of 2 rhs
template < typename L, typename R >
constexpr auto power_of_2_mod(L lhs, R rhs) noexcept
{
  return lhs & (rhs-1);
}

template < size_t value >
struct ConstantModel
{
  static constexpr size_t get_replication(size_t RAJAPERF_UNUSED_ARG(parallelism)) noexcept
  {
    return value;
  }
};

template < size_t cutoff, size_t value_before_cutoff, size_t value_after_cutoff >
struct CutoffModel
{
  static constexpr size_t get_replication(size_t parallelism) noexcept
  {
    return parallelism <= cutoff ? value_before_cutoff : value_after_cutoff;
  }
};

template < typename T, typename IndexType >
struct MultiReduceAtomicCalculator
{
  template < typename SharedAtomicModel >
  static constexpr IndexType calculate_shared_replication(IndexType num_bins,
                                                          IndexType threads_per_block,
                                                          IndexType max_shmem_per_block_in_bytes,
                                                          SharedAtomicModel shared_atomic_model)
  {
    IndexType shared_replication = shared_atomic_model.get_replication(threads_per_block);
    IndexType max_shared_replication = max_shmem_per_block_in_bytes / sizeof(T) / num_bins;
    return prev_pow2(std::min(shared_replication, max_shared_replication));
  }

  template < typename GlobalAtomicModel >
  static constexpr IndexType calculate_global_replication(IndexType threads_per_block,
                                                           IndexType blocks_per_kernel,
                                                           GlobalAtomicModel global_atomic_model)
  {
    IndexType global_replication = global_atomic_model.get_replication(threads_per_block);
    return next_pow2(std::min(global_replication, blocks_per_kernel));
  }

  template < typename GlobalAtomicModel, typename SharedAtomicModel >
  constexpr MultiReduceAtomicCalculator(IndexType num_bins,
                                        IndexType threads_per_block,
                                        IndexType blocks_per_kernel,
                                        IndexType max_shmem_per_block_in_bytes,
                                        GlobalAtomicModel global_atomic_model,
                                        SharedAtomicModel shared_atomic_model)
    : m_num_bins(num_bins)
    , m_shared_replication(calculate_shared_replication(num_bins, threads_per_block, max_shmem_per_block_in_bytes, shared_atomic_model))
    , m_global_replication(calculate_global_replication(threads_per_block, blocks_per_kernel, global_atomic_model))
  { }

  // get the shared memory usage in bytes
  __host__ __device__
  constexpr IndexType shared_memory_in_bytes() const
  {
    return m_shared_replication * m_num_bins * sizeof(T);
  }

  // get the number of bins
  __host__ __device__
  constexpr IndexType num_bins() const
  {
    return m_num_bins;
  }

  // get the shared replication, always a power of 2
  __host__ __device__
  constexpr IndexType shared_replication() const
  {
    return m_shared_replication;
  }

  // get the global replication, always a power of 2
  __host__ __device__
  constexpr IndexType global_replication() const
  {
    return m_global_replication;
  }

  // get the offset into shared memory
  __host__ __device__
  constexpr IndexType get_shared_offset(IndexType bin, IndexType rep) const
  {
    // make rep stride-1 to avoid bank conflicts
    return bin * shared_replication() + power_of_2_mod(rep, shared_replication());
  }

  // get the offset into global memory
  __host__ __device__
  constexpr IndexType get_global_offset(IndexType bin, IndexType rep) const
  {
    // make bin stride-1 so atomics from a single block can coalesce
    return bin + power_of_2_mod(rep, global_replication()) * num_bins();
  }

  template < typename IterFinal, typename IterGlobal, typename Op >
  T combine_global(IndexType bin, IterGlobal counts_global, Op combiner)
  {
    T count_final = combiner.identity();
    for (IndexType rep = 0; rep < global_replication(); ++rep) {
      combiner(count_final, counts_global[get_global_offset(bin, rep)]);
    }
    return count_final;
  }

  template < typename IterFinal, typename IterGlobal, typename Op >
  void combine_globals(IterFinal counts_final, IterGlobal counts_global, Op combiner)
  {
    for (IndexType bin = 0; bin < num_bins; ++bin) {
      counts_final[bin] = combine_global(bin, counts_global, combiner);
    }
  }

private:
  IndexType m_num_bins;
  IndexType m_shared_replication;
  IndexType m_global_replication;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
