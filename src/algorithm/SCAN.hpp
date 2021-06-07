//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
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
  y[ibegin] = 0.0;

#define SCAN_BODY \
  y[i] = y[i-1] + x[i-1];

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

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid)
  {
    std::cout << "\n  SCAN : Unknown OMP Target variant id = " << vid << std::endl;
  }

private:
  Real_ptr m_x;
  Real_ptr m_y;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
