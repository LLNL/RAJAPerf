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
/// POLYBENCH_2MM kernel reference implementation:
///
/// D := alpha*A*B*C + beta*D
///
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type j = 0; j < nj; j++) {
///     tmp[i][j] = 0.0;
///     for (Index_type k = 0; k < nk; ++k) {
///       tmp[i][j] += alpha * A[i][k] * B[k][j];
///     }
///   }
/// } 
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type l = 0; l < nl; l++) {
///     D[i][l] *= beta;  // NOTE: Changed to 'D[i][l] = beta;' 
///                       // to avoid need for memset operation
///                       // to zero out matrix.
///     for (Index_type j = 0; j < nj; ++j) {
///       D[i][l] += tmp[i][j] * C[j][l];
///     } 
///   }
/// } 
///



#ifndef RAJAPerf_POLYBENCH_2MM_HPP
#define RAJAPerf_POLYBENCH_2MM_HPP

#define POLYBENCH_2MM_BODY1 \
  tmp[j + i*nj] = 0.0;

#define POLYBENCH_2MM_BODY2 \
  tmp[j + i*nj] += alpha * A[k + i*nk] * B[j + k*nj];

#define POLYBENCH_2MM_BODY3 \
  D[l + i*nl] = beta;

#define POLYBENCH_2MM_BODY4 \
  D[l + i*nl] += tmp[j + i*nj] * C[l + j*nl];


#define POLYBENCH_2MM_BODY1_RAJA \
  tmpview(i,j) = 0.0;

#define POLYBENCH_2MM_BODY2_RAJA \
  tmpview(i,j) += alpha * Aview(i,k) * Bview(k,j);

#define POLYBENCH_2MM_BODY3_RAJA \
  Dview(i,l) = beta;

#define POLYBENCH_2MM_BODY4_RAJA \
  Dview(i,l) += tmpview(i,j) * Cview(j, l);

#define POLYBENCH_2MM_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE tmpview(tmp, RAJA::Layout<2>(ni, nj)); \
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(nj, nl)); \
  VIEW_TYPE Dview(D, RAJA::Layout<2>(ni, nl));


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
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_tmp;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
