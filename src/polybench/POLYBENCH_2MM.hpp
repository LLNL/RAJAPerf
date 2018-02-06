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
/// POLYBENCH_2MM kernel reference implementation:
///
/// D := alpha*A*B*C + beta*D
///
/// for (Index_type i = 0; i < m_ni; i++) {
///   for (Index_type j = 0; j < m_nj; j++) {
///     m_tmp[i][j] = 0.0;
///     for (Index_type k = 0; k < m_nk; ++k) {
///       m_tmp[i][j] += m_alpha * m_A[i][k] * m_B[k][j];
///     }
///   }
/// } 
/// for (Index_type i = 0; i < m_ni; i++) {
///   for (Index_type j = 0; j < m_nl; j++) {
///     m_D[i][j] *= m_beta;
///     for (Index_type k = 0; k < m_nj; ++k) {
///       m_D[i][j] += m_tmp[i][k] * m_C[k][j];
///     } 
///   }
/// } 
///



#ifndef RAJAPerf_POLYBENCH_2MM_HXX
#define RAJAPerf_POLYBENCH_2MM_HXX

#define POLYBENCH_2MM_BODY1 \
  *(tmp + i * nj + j) = 0.0;

#define POLYBENCH_2MM_BODY2 \
  *(tmp + i * nj + j) += alpha * *(A + i * nk + k) * *(B + k * nj + j);

#define POLYBENCH_2MM_BODY3 \
  *(D + i * nl + l) *= beta;

#define POLYBENCH_2MM_BODY4 \
  *(D + i * nl + l) += *(tmp + i * nj + j) * *(C + j * nl + l);

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_2MM : public KernelBase
{
public:

  POLYBENCH_2MM(const RunParams& params);

  ~POLYBENCH_2MM();


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
  Index_type m_run_reps;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_tmp;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
  Real_ptr m_DD;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
