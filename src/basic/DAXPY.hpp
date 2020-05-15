//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DAXPY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] += a * x[i] ;
/// }
///

#ifndef RAJAPerf_Basic_DAXPY_HPP
#define RAJAPerf_Basic_DAXPY_HPP

#define DAXPY_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_type a = m_a; \
  RAJA_INDEX_VALUE_T(I, int, "I");\
  using vector_t = RAJA::StreamVector<Real_type, 2>;\
  using VecI = RAJA::VectorIndex<I, vector_t>;\
  RAJA::TypedView<Real_type, RAJA::Layout<1>, I> X(x, getRunSize()); \
  RAJA::TypedView<Real_type, RAJA::Layout<1>, I> Y(y, getRunSize()); \


#define DAXPY_BODY  \
  y[i] += a * x[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class DAXPY : public KernelBase
{
public:

  DAXPY(const RunParams& params);

  ~DAXPY();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_type m_a;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
