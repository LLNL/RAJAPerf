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
  const Real_type a = m_a;

#define DAXPY_BODY  \
  y[i] += a * x[i] ;

#define DAXPY_DATA_VEC_SETUP \
  RAJA_INDEX_VALUE_T(I, Int_type, "I");\
  using vector_t = RAJA::StreamVector<Real_type, 2>; \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> X(x, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> Y(y, iend);

#define DAXPY_DATA_VEC_SETUP2 \
  RAJA_INDEX_VALUE_T(I, Int_type, "I"); \
  using vector_t = RAJA::StreamVector<Real_type,2>; \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> Xview(x, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> Yview(y, iend); \
  RAJA::forall<RAJA::vector_exec<vector_t>> (RAJA::TypedRangeSegment<I>(ibegin, iend),\
  [=](RAJA::VectorIndex<I, vector_t> i) { \
    vector_t X(0), Y(0); \
    for(int j = 0; j < i.size(); ++j) { \
      X.set(j, *(x + (**i) + j)); \
      Y.set(j, *(y + (**i) + j)); \
    } \
    Xview(i) = X; \
    Yview(i) = Y; \
  });

#define DAXPY_DATA_VEC_SETUP3 \
  RAJA_INDEX_VALUE_T(I, Int_type, "I");\
  using element_t = RAJA::StreamVector<Real_type,2>::element_type; \
  element_t X[iend], Y[iend]; \
  for(int i = 0; i < iend; ++i) { \
    X[i] = x[i]; \
    Y[i] = y[i]; \
  }

#define DAXPY_VEC_BODY \
  Y(i) += a * X(i);

#define DAXPY_VEC_BODY2 \
  Yview(i) += a*Xview(i);

#define DAXPY_VEC_BODY3 \
  for(int i = 0;i < iend; ++i){ \
    Y[i] += a * X[i]; \
    y[i] = Y[i]; \
  }

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
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_type m_a;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
