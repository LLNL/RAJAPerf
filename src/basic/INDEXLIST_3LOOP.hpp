//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INDEXLIST_3LOOP kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   counts[i] = (x[i] < 0.0) ? 1 : 0;
/// }
///
/// Idx_type count = 0;
/// for (Index_type i = ibegin; i < iend+1; ++i ) {
///   Idx_type inc = counts[i];
///   counts[i] = count;
///   count += inc;
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if (counts[i] != counts[i+1]) {
///     list[counts[i]] = i;
///   }
/// }
///
/// Idx_type len = counts[iend];
///

#ifndef RAJAPerf_Basic_INDEXLIST_3LOOP_HPP
#define RAJAPerf_Basic_INDEXLIST_3LOOP_HPP

#define INDEXLIST_3LOOP_DATA_SETUP \
  Real_ptr x = m_x; \
  Idx_ptr list = m_list;

#define INDEXLIST_3LOOP_CONDITIONAL \
  x[i] < 0.0

#define INDEXLIST_3LOOP_MAKE_LIST \
  if (counts[i] != counts[i+1]) { \
    list[counts[i]] = i ; \
  }


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class INDEXLIST_3LOOP : public KernelBase
{
public:
  using Idx_type = Index_type;
  using Idx_ptr = Index_ptr;

  INDEXLIST_3LOOP(const RunParams& params);

  ~INDEXLIST_3LOOP();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Idx_ptr m_list;
  Idx_type m_len;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
