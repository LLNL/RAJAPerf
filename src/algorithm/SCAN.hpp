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
  Real_ptr x = m_x; \
  Real_ptr y = m_y;

#define SCAN_PROLOGUE \
  Real_type scan_var = 0.0;

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

private:
  static const size_t default_gpu_block_size = 0;

  Real_ptr m_x;
  Real_ptr m_y;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
