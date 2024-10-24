//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ATOMIC kernel reference implementation:
/// Test atomic throughput with an amount of replication known at compile time.
///
/// for (Index_type i = 0; i < N; ++i ) {
///   atomic[i%replication] += 1;
/// }
///

#ifndef RAJAPerf_Algorithm_ATOMIC_HPP
#define RAJAPerf_Algorithm_ATOMIC_HPP

#define ATOMIC_DATA_SETUP(replication) \
  Real_type init = m_init; \
  Real_ptr atomic; \
  allocAndInitDataConst(atomic, replication, init, vid);

#define ATOMIC_DATA_TEARDOWN(replication) \
  { \
    auto reset_atomic = scopedMoveData(atomic, replication, vid); \
    m_final = init; \
    for (size_t r = 0; r < replication; ++r ) { \
      m_final += atomic[r]; \
    } \
  } \
  deallocData(atomic, vid);

#define ATOMIC_VALUE 1.0

#define ATOMIC_BODY(i, val) \
  atomic[(i)%replication] += (val)

#define ATOMIC_RAJA_BODY(policy, i, val) \
  RAJA::atomicAdd<policy>(&atomic[(i)%replication], (val))


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class ATOMIC : public KernelBase
{
public:

  ATOMIC(const RunParams& params);

  ~ATOMIC();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runKokkosVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void setOpenMPTuningDefinitions(VariantID vid);
  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setOpenMPTargetTuningDefinitions(VariantID vid);

  template < size_t replication >
  void runSeqVariantReplicate(VariantID vid);

  template < size_t replication >
  void runOpenMPVariantReplicate(VariantID vid);

  template < size_t block_size, size_t replication >
  void runCudaVariantReplicateGlobal(VariantID vid);
  template < size_t block_size, size_t replication >
  void runHipVariantReplicateGlobal(VariantID vid);
  template < size_t block_size, size_t replication >

  void runCudaVariantReplicateWarp(VariantID vid);
  template < size_t block_size, size_t replication >
  void runHipVariantReplicateWarp(VariantID vid);

  template < size_t block_size, size_t replication >
  void runCudaVariantReplicateBlock(VariantID vid);
  template < size_t block_size, size_t replication >
  void runHipVariantReplicateBlock(VariantID vid);

  template < size_t replication >
  void runOpenMPTargetVariantReplicate(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;
  static const size_t default_cpu_atomic_replication = 64;
  using cpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_cpu_atomic_replication>;
  static const size_t default_atomic_replication = 4096;
  using gpu_atomic_replications_type = integer::make_atomic_replication_list_type<default_atomic_replication>;

  Real_type m_init;
  Real_type m_final;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
