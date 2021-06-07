//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INDEXLIST kernel reference implementation:
///
/// Index_type count = 0;
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if (x[i] < 0.0) { \
///     list[count++] = i ; \
///   }
/// }
/// Index_type len = count;
///

#ifndef RAJAPerf_Basic_INDEXLIST_HPP
#define RAJAPerf_Basic_INDEXLIST_HPP

#define INDEXLIST_DATA_SETUP \
  Real_ptr x = m_x; \
  Int_ptr list = m_list;

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

  INDEXLIST(const RunParams& params);

  ~INDEXLIST();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Int_ptr m_list;
  Index_type m_len;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
