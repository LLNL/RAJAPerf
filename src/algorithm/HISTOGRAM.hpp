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
  std::vector<Data_type>& counts_init = m_counts_init; \
  std::vector<Data_type>& counts_final = m_counts_final;

#define HISTOGRAM_DATA_TEARDOWN


#define HISTOGRAM_SETUP_COUNTS \
  Data_ptr counts; \
  allocData(getReductionDataSpace(vid), counts, num_bins);

#define HISTOGRAM_TEARDOWN_COUNTS \
  deallocData(counts, vid);

#define HISTOGRAM_INIT_COUNTS \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    counts[b] = counts_init[b]; \
  }

#define HISTOGRAM_FINALIZE_COUNTS \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    counts_final[b] = counts[b]; \
  }

#define HISTOGRAM_INIT_COUNTS_RAJA(policy) \
  RAJA::MultiReduceSum<policy, Data_type> counts(counts_init);

#define HISTOGRAM_FINALIZE_COUNTS_RAJA(policy) \
  counts.get_all(counts_final);

#define HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, replication) \
  for (Index_type b = 0; b < (num_bins); ++b) { \
    Data_type count_final = 0; \
    for (size_t r = 0; r < (replication); ++r) { \
      count_final += (hcounts)[HISTOGRAM_GPU_BIN_INDEX(b, r, replication)]; \
    } \
    counts_final[b] = count_final; \
  }


#define HISTOGRAM_BODY \
  counts[bins[i]] += static_cast<Data_type>(1);

#define HISTOGRAM_RAJA_BODY(policy) \
  RAJA::atomicAdd<policy>(&counts[bins[i]], static_cast<Data_type>(1));

#define HISTOGRAM_GPU_BIN_INDEX(bin, offset, replication) \
  ((bin)*(replication) + ((offset)%(replication)))

#define HISTOGRAM_GPU_RAJA_BODY(policy, counts, index, value) \
  RAJA::atomicAdd<policy>(&(counts)[(index)], (value));


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

  template < Index_type block_size,
             Index_type preferred_global_replication,
             Index_type preferred_shared_replication,
             typename MappingHelper >
  void runCudaVariantAtomicRuntime(VariantID vid);
  template < Index_type block_size,
             Index_type preferred_global_replication,
             Index_type preferred_shared_replication,
             typename MappingHelper >
  void runHipVariantAtomicRuntime(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  static const size_t default_cuda_atomic_global_replication = 2;
  static const size_t default_cuda_atomic_shared_replication = 16;
  using cuda_atomic_global_replications_type = integer::make_atomic_replication_list_type<0>; // default list is empty
  using cuda_atomic_shared_replications_type = integer::make_atomic_replication_list_type<0>; // default list is empty

  static const size_t default_hip_atomic_global_replication = 32;
  static const size_t default_hip_atomic_shared_replication = 4;
  using hip_atomic_global_replications_type = integer::make_atomic_replication_list_type<0>; // default list is empty
  using hip_atomic_shared_replications_type = integer::make_atomic_replication_list_type<0>; // default list is empty

  Index_type m_num_bins;
  Index_ptr m_bins;
  std::vector<Data_type> m_counts_init;
  std::vector<Data_type> m_counts_final;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
