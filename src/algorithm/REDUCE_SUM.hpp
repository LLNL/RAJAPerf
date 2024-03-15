//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// REDUCE_SUM kernel reference implementation:
///
/// Data_type sum = std::reduce(x+ibegin, x+iend);
/// // or
/// Data_type sum = std::accumulate(x+ibegin, x+iend, 0.0);
/// // or
/// Data_type sum = 0;
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   sum += x[i] ;
/// }
///

#ifndef RAJAPerf_Algorithm_REDUCE_SUM_HPP
#define RAJAPerf_Algorithm_REDUCE_SUM_HPP

#define REDUCE_SUM_DATA_SETUP \
  Data_ptr x = m_x;

#define REDUCE_SUM_STD_ARGS  \
  x + ibegin, x + iend

#define REDUCE_SUM_VAL \
  x[i]

#define REDUCE_SUM_OP(sum, val) \
  (sum) += (val)

#define REDUCE_SUM_VAR_BODY(sum) \
  REDUCE_SUM_OP(sum, REDUCE_SUM_VAL)

#define REDUCE_SUM_BODY \
  REDUCE_SUM_VAR_BODY(sum)


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class REDUCE_SUM : public KernelBase
{
public:
  using Data_type = Real_type;
  using Data_ptr = Real_ptr;

  REDUCE_SUM(const RunParams& params);

  ~REDUCE_SUM();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void runSeqVariantBinary(VariantID vid);
  void runSeqVariantDefault(VariantID vid);
  template < size_t replication >
  void runSeqVariantReplication(VariantID vid);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void runCudaVariantCub(VariantID vid);
  void runHipVariantRocprim(VariantID vid);
  template < size_t block_size, typename MappingHelper, typename AtomicOrdering >
  void runCudaVariantBase(VariantID vid, AtomicOrdering atomic_ordering);
  template < size_t block_size, typename MappingHelper, typename AtomicOrdering >
  void runHipVariantBase(VariantID vid, AtomicOrdering atomic_ordering);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runCudaVariantRAJA(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runHipVariantRAJA(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;
  static const size_t default_cpu_atomic_replication = 64;
  using cpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_cpu_atomic_replication>;
  static const size_t default_gpu_atomic_replication = 4096; // 1024, 8192
  using gpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_gpu_atomic_replication>;


  Data_ptr m_x;
  Data_type m_sum_init;
  Data_type m_sum;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
