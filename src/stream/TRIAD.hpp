//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRIAD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = b[i] + alpha * c[i] ;
/// }
///

#ifndef RAJAPerf_Stream_TRIAD_HPP
#define RAJAPerf_Stream_TRIAD_HPP

#define TRIAD_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_type alpha = m_alpha;

#define TRIAD_DATA_VEC_SETUP \
  RAJA_INDEX_VALUE_T(I, Int_type, "I"); \
  using vector_t = RAJA::StreamVector<Real_type, 2>; \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type>, I> A(a, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type>, I> B(b, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type>, I> C(c, iend); 

#define TRIAD_BODY  \
  a[i] = b[i] + alpha * c[i] ;

#define TRIAD_VEC_BODY \
 A(i) = B(i) + alpha * C(i);

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class TRIAD : public KernelBase
{
public:

  TRIAD(const RunParams& params);

  ~TRIAD();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
