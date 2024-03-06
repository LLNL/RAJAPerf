//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SCAN kernel reference implementation:
///
/// // exclusive scan
/// y[ibegin] = 0;
/// for (Index_type i = ibegin+1; i < iend; ++i) {
///   y[i] = y[i-1] + x[i-1];
/// }
///

#ifndef RAJAPerf_Algorithm_SCAN_HPP
#define RAJAPerf_Algorithm_SCAN_HPP

#define SCAN_DATA_SETUP \
  Data_ptr x = m_x; \
  Data_ptr y = m_y;

#define SCAN_PROLOGUE \
  Data_type scan_var = 0;

#define SCAN_BODY \
  y[i] = scan_var; \
  scan_var += x[i];

#define RAJA_SCAN_ARGS \
  RAJA::make_span(x + ibegin, iend - ibegin), \
  RAJA::make_span(y + ibegin, iend - ibegin)


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class SCAN : public KernelBase
{
public:
  using Data_type = Real_type;
  using Data_ptr = Real_ptr;

  SCAN(const RunParams& params);

  ~SCAN();

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
  void runCudaVariantLibrary(VariantID vid);
  void runHipVariantLibrary(VariantID vid);
  template < size_t block_size, size_t items_per_thread >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size, size_t items_per_thread >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Data_ptr m_x;
  Data_ptr m_y;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
