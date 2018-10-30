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
/// POLYBENCH_GEMMVER kernel reference implementation:
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
///   }
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     x[i] = x[i] + beta * A[j][i] * y[j];
///   }
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   x[i] = x[i] + z[i];
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     w[i] = w[i] +  alpha * A[i][j] * x[j];
///   }
/// }
///



#ifndef RAJAPerf_POLYBENCH_GEMMVER_HPP
#define RAJAPerf_POLYBENCH_GEMMVER_HPP

#define POLYBENCH_GEMMVER_BODY1 \
  A[j + i*n] += u1[i] * v1[j] + u2[i] * v2[j];

#define POLYBENCH_GEMMVER_BODY2 \
  x[i] +=  beta * A[i + j*n] * y[j];

#define POLYBENCH_GEMMVER_BODY3 \
  x[i] += z[i];

#define POLYBENCH_GEMMVER_BODY4 \
  w[i] +=  alpha * A[j + i*n] * x[j];


#define POLYBENCH_GEMMVER_BODY1_RAJA \
  Aview(i,j) += u1view(i) * v1view(j) + u2view(i) * v2view(j);

#define POLYBENCH_GEMMVER_BODY2_RAJA \
  xview(i) +=  beta * Aview(j,i) * yview(j);

#define POLYBENCH_GEMMVER_BODY3_RAJA \
  xview(i) += zview(i);

#define POLYBENCH_GEMMVER_BODY4_RAJA \
  wview(i) +=  alpha * Aview(i,j) * xview(j);

#define POLYBENCH_GEMMVER_VIEWS_RAJA \
  using VIEW_1 = RAJA::View<Real_type, \
                            RAJA::Layout<1, Index_type, 0>>; \
\
  using VIEW_2 = RAJA::View<Real_type, \
                            RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_1 u1view(u1, RAJA::Layout<1>(n)); \
  VIEW_1 v1view(v1, RAJA::Layout<1>(n)); \
  VIEW_1 u2view(u2, RAJA::Layout<1>(n)); \
  VIEW_1 v2view(v2, RAJA::Layout<1>(n)); \
  VIEW_1 wview(w, RAJA::Layout<1>(n)); \
  VIEW_1 xview(x, RAJA::Layout<1>(n)); \
  VIEW_1 yview(y, RAJA::Layout<1>(n)); \
  VIEW_1 zview(z, RAJA::Layout<1>(n)); \
  VIEW_2 Aview(A, RAJA::Layout<2>(n, n));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMMVER : public KernelBase
{
public:

  POLYBENCH_GEMMVER(const RunParams& params);

  ~POLYBENCH_GEMMVER();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_n;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_A;
  Real_ptr m_u1;
  Real_ptr m_v1;
  Real_ptr m_u2;
  Real_ptr m_v2;
  Real_ptr m_w;
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
