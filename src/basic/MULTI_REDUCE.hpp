//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// MULTI_REDUCE kernel reference implementation:
///
/// double* values = calloc(num_bins, sizeof(double));
/// for (Index_type i = 0; i < N; ++i ) {
///   values[bins[i]] += data[i];
/// }
///

#ifndef RAJAPerf_Basic_MULTI_REDUCE_HPP
#define RAJAPerf_Basic_MULTI_REDUCE_HPP

#define MULTI_REDUCE_DATA_SETUP \
  Index_type num_bins = m_num_bins; \
  Index_ptr bins = m_bins; \
  Data_ptr data = m_data; \
  Data_ptr values_init = m_values_init.data(); \
  Data_ptr values_final = m_values_final.data(); \
  Data_ptr values; \
  allocData(getReductionDataSpace(vid), values, num_bins);

#define MULTI_REDUCE_DATA_TEARDOWN \
  deallocData(values, vid);

#define MULTI_REDUCE_GPU_DATA_SETUP \
  Index_type num_bins = m_num_bins; \
  Index_ptr bins = m_bins; \
  Data_ptr data = m_data; \
  Data_ptr values_init = m_values_init.data(); \
  Data_ptr values_final = m_values_final.data();

#define MULTI_REDUCE_BODY \
  values[bins[i]] += data[i];

#define MULTI_REDUCE_RAJA_BODY(policy) \
  RAJA::atomicAdd<policy>(&values[bins[i]], data[i]);

#define MULTI_REDUCE_GPU_RAJA_BODY(policy) \
  RAJA::atomicAdd<policy>(&values[bins[i]*replication + (i%replication)], data[i]);

#define MULTI_REDUCE_INIT_VALUES \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    values[b] = values_init[b]; \
  }

#define MULTI_REDUCE_FINALIZE_VALUES \
  for (Index_type b = 0; b < num_bins; ++b ) { \
    values_final[b] = values[b]; \
  }

#define MULTI_REDUCE_GPU_FINALIZE_VALUES(hvalues, num_bins, replication) \
  for (Index_type b = 0; b < (num_bins); ++b) { \
    Data_type val_final = 0; \
    for (size_t r = 0; r < (replication); ++r) { \
      val_final += (hvalues)[b*(replication) + r]; \
    } \
    values_final[b] = val_final; \
  }


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class MULTI_REDUCE : public KernelBase
{
public:
  using Data_type = Real_type;
  using Data_ptr = Real_ptr;

  MULTI_REDUCE(const RunParams& params);

  ~MULTI_REDUCE();

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
  template < size_t block_size, size_t replication >
  void runCudaVariantAtomicGlobal(VariantID vid);
  template < size_t block_size, size_t replication >
  void runHipVariantAtomicGlobal(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;
  static const size_t default_gpu_atomic_replication = 2048; // 512, 512
  using gpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_gpu_atomic_replication>;

  Index_type m_num_bins;
  Index_ptr m_bins;
  Data_ptr m_data;
  std::vector<Data_type> m_values_init;
  std::vector<Data_type> m_values_final;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
