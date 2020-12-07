//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FIRST_SUM kernel reference implementation:
///
/// Note: kernel is altered to enable parallelism (original used 'x[i-1]'
///       on the right-hand side).
///
/// for (Index_type i = 1; i < N; ++i ) {
///   x[i] = y[i-1] + y[i];
/// }
///

#ifndef RAJAPerf_Lcals_FIRST_SUM_HPP
#define RAJAPerf_Lcals_FIRST_SUM_HPP


#define FIRST_SUM_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y;

#define FIRST_SUM_VEC_SETUP \
  RAJA_INDEX_VALUE_T(I, Int_type, "I"); \
  using element_t = RAJA::StreamVector<Real_type,2>::element_type; \
  element_t X[iend], Y[iend]; \
  for(int i = 0; i < iend; ++i) { \
    X[i] = x[i]; \
    Y[i] = y[i]; \
  }

#define FIRST_SUM_VEC_BODY \
  for(int i = 1; i < iend; ++i) { \
    X[i] = Y[i-1] + Y[i]; \
    x[i] = X[i]; \
  }

#define FIRST_SUM_BODY  \
  x[i] = y[i-1] + y[i];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class FIRST_SUM : public KernelBase
{
public:

  FIRST_SUM(const RunParams& params);

  ~FIRST_SUM();

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
  Real_ptr m_y;

  Index_type m_N;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
