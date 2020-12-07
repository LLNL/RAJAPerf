//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// COPY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   c[i] = a[i] ;
/// }
///

#ifndef RAJAPerf_Stream_COPY_HPP
#define RAJAPerf_Stream_COPY_HPP

#define COPY_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr c = m_c;

#define COPY_DATA_VEC_SETUP \
  RAJA_INDEX_VALUE_T(I, Int_type, "I"); \
  using vector_t = RAJA::StreamVector<Real_type, 2>; \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> A(a, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> C(c, iend); 

#define COPY_BODY  \
  c[i] = a[i] ;

#define COPY_VEC_BODY \
 C(i) = A(i);

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class COPY : public KernelBase
{
public:

  COPY(const RunParams& params);

  ~COPY();

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
  Real_ptr m_c;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
