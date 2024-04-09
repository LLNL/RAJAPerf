//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DOT kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   dot += a[i] * b[i];
/// }
///

#ifndef RAJAPerf_Stream_DOT_HPP
#define RAJAPerf_Stream_DOT_HPP

#define DOT_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr b = m_b;

#define DOT_BODY  \
  dot += a[i] * b[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace stream
{

class DOT : public KernelBase
{
public:

  DOT(const RunParams& params);

  ~DOT();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void runKokkosVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid);

  template < size_t block_size, typename MappingHelper >
  void runCudaVariantBase(VariantID vid);
  template < size_t block_size, typename MappingHelper >
  void runHipVariantBase(VariantID vid);

  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runCudaVariantRAJA(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runHipVariantRAJA(VariantID vid);

  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_a;
  Real_ptr m_b;
  Real_type m_dot;
  Real_type m_dot_init;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
