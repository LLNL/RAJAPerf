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

#define HISTOGRAM_GPU_RAJA_BODY(policy) \
  RAJA::atomicAdd<policy>(&counts[bins[i]*replication + (i%replication)], static_cast<HISTOGRAM::Data_type>(1));

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
      count_final += (hcounts)[b*(replication) + r]; \
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
  template < size_t block_size, size_t replication >
  void runCudaVariantReplicateGlobal(VariantID vid);
  template < size_t block_size, size_t replication >
  void runHipVariantReplicateGlobal(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;
  static const size_t default_atomic_replication = 4096;
  using gpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_atomic_replication>;

  Index_type m_num_bins;
  Index_ptr m_bins;
  std::vector<Data_type> m_counts_init;
  std::vector<Data_type> m_counts_final;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
