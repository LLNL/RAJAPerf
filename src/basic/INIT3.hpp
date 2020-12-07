//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT3 kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;
/// }
///

#ifndef RAJAPerf_Basic_INIT3_HPP
#define RAJAPerf_Basic_INIT3_HPP


#define INIT3_DATA_SETUP \
  Real_ptr out1 = m_out1; \
  Real_ptr out2 = m_out2; \
  Real_ptr out3 = m_out3; \
  Real_ptr in1 = m_in1; \
  Real_ptr in2 = m_in2;

#define INIT3_VEC_SETUP \
  RAJA_INDEX_VALUE_T(I, Int_type, "I"); \
  using vector_t = RAJA::StreamVector<Real_type,2>; \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> O1(out1, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> O2(out2, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> O3(out3, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> I1(in1, iend); \
  RAJA::TypedView<Real_type, RAJA::Layout<1, Int_type, 0>, I> I2(in2, iend);

#define INIT3_BODY  \
  out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;

#define INIT3_VEC_BODY \
  O1(i) = O2(i) = O3(i) = -1 * I1(i) - I2(i);

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class INIT3 : public KernelBase
{
public:

  INIT3(const RunParams& params);

  ~INIT3();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_out1;
  Real_ptr m_out2;
  Real_ptr m_out3;
  Real_ptr m_in1;
  Real_ptr m_in2; 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
