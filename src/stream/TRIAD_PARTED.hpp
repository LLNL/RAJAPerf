//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRIAD_PARTED kernel reference implementation:
///
/// for (size_t p = 1; p < parts.size(); ++p ) {
///   Index_type ibegin = iparts[p-1];
///   Index_type iend = iparts[p];
///   for (Index_type i = ibegin; i < iend; ++i ) {
///     a[i] = b[i] + alpha * c[i] ;
///   }
/// }
///

#ifndef RAJAPerf_Stream_TRIAD_PARTED_HPP
#define RAJAPerf_Stream_TRIAD_PARTED_HPP

#define TRIAD_PARTED_DATA_SETUP \
  std::vector<Index_type> parts = m_parts; \
  \
  Real_ptr a = m_a; \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_type alpha = m_alpha;

#define TRIAD_PARTED_BODY  \
  a[i] = b[i] + alpha * c[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace stream
{

class TRIAD_PARTED : public KernelBase
{
public:

  TRIAD_PARTED(const RunParams& params);

  ~TRIAD_PARTED();

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
  template < size_t block_size >
  void runCudaVariantBlock(VariantID vid);
  template < size_t block_size >
  void runCudaVariantOpenmp(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantBlock(VariantID vid);
  template < size_t block_size >
  void runHipVariantOpenmp(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  std::vector<Index_type> m_parts;

  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
