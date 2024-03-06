//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INDEXLIST kernel reference implementation:
///
/// Idx_type count = 0;
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if (x[i] < 0.0) {
///     list[count++] = i ;
///   }
/// }
/// Idx_type len = count;
///

#ifndef RAJAPerf_Basic_INDEXLIST_HPP
#define RAJAPerf_Basic_INDEXLIST_HPP

#define INDEXLIST_DATA_SETUP \
  Real_ptr x = m_x; \
  Idx_ptr list = m_list;

#define INDEXLIST_CONDITIONAL  \
  x[i] < 0.0

#define INDEXLIST_BODY  \
  if (INDEXLIST_CONDITIONAL) { \
    list[count++] = i ; \
  }


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class INDEXLIST : public KernelBase
{
public:
  using Idx_type = Index_type;
  using Idx_ptr = Index_ptr;

  INDEXLIST(const RunParams& params);

  ~INDEXLIST();

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
  template < size_t block_size, size_t items_per_thread >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size, size_t items_per_thread >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Idx_ptr m_list;
  Idx_type m_len;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
