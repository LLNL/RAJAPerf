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
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
///
/// POLYBENCH_GEMM kernel reference implementation:
///
/// for (Index_type i = 0; i < NI; i++) {
///   for (Index_type j = 0; j < NJ; j++) {
///     C[i][j] *= beta;
///     double dot = 0.0;
///     for (Index_type k = 0; k < NK; k++) {
///       dot += alpha * A[i][k] * B[k][j];
///     }
///     C[i][j] = dot;
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_GEMM_HPP
#define RAJAPerf_POLYBENCH_GEMM_HPP

#define POLYBENCH_GEMM_BODY1 \
  C[j + i*nj] *= beta; \
  double dot = 0.0;

#define POLYBENCH_GEMM_BODY2 \
  dot += alpha * A[k + i*nk] * B[j + k*nj];  

#define POLYBENCH_GEMM_BODY3 \
  C[j + i*nj] = dot;


#define POLYBENCH_GEMM_BODY1_RAJA \
  Cview(i, j) *= beta; \
  dot = 0.0;

#define POLYBENCH_GEMM_BODY2_RAJA \
  dot += alpha * Aview(i, k) * Bview(k, j);  

#define POLYBENCH_GEMM_BODY3_RAJA \
  Cview(i, j) = dot;

#define POLYBENCH_GEMM_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, \
                               RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(ni, nj));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMM : public KernelBase
{
public:

  POLYBENCH_GEMM(const RunParams& params);

  ~POLYBENCH_GEMM();


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

  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
