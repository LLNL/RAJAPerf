//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
///
/// POLYBENCH_3MM kernel reference implementation:
///
/// E := A*B 
/// F := C*D 
/// G := E*F 
///
/// for (Index_type i = 0; i < _PB_NI; i++) {
///   for (Index_type j = 0; j < _PB_NJ; j++) {
///     E[i][j] = SCALAR_VAL(0.0);
///     for (Index_type k = 0; k < _PB_NK; ++k) {
///       E[i][j] += A[i][k] * B[k][j];
///     }
///   }
/// } 
/// for (Index_type i = 0; i < _PB_NJ; i++) {
///   for (Index_type j = 0; j < _PB_NL; j++) {
///	F[i][j] = SCALAR_VAL(0.0);
///	for (Index_type k = 0; k < _PB_NM; ++k) {
///	  F[i][j] += C[i][k] * D[k][j];
///     }
///   }
/// }
/// for (Index_type i = 0; i < _PB_NI; i++) {
///   for (Index_type j = 0; j < _PB_NL; j++) {
///     G[i][j] = SCALAR_VAL(0.0);
///     for (Index_type k = 0; k < _PB_NJ; ++k) {
///	  G[i][j] += E[i][k] * F[k][j];
///     }
///   }
/// }
///

#ifndef RAJAPerf_POLYBENCH_3MM_HXX
#define RAJAPerf_POLYBENCH_3MM_HXX

#define POLYBENCH_3MM_BODY1 \
  *(E + i * nj + j) = 0.0;

#define POLYBENCH_3MM_BODY2 \
  *(E + i * nj + j) += *(A + i * nk + k) * *(B + k * nj + j);

#define POLYBENCH_3MM_BODY3 \
  *(F + j * nl + l) = 0.0;

#define POLYBENCH_3MM_BODY4 \
  *(F + j * nl + l)  += *(C + j * nm + m) * *(D + m * nl + l);

#define POLYBENCH_3MM_BODY5 \
  *(G + i * nl + l) = 0.0;

#define POLYBENCH_3MM_BODY6 \
  *(G + i * nl + l) += *(E + i * nj + j) * *(F + j * nl + l);


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_3MM : public KernelBase
{
public:

  POLYBENCH_3MM(const RunParams& params);

  ~POLYBENCH_3MM();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nl;
  Index_type m_nm;
  Index_type m_run_reps;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
  Real_ptr m_E;
  Real_ptr m_F;
  Real_ptr m_G;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
