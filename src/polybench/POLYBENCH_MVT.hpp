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
/// POLYBENCH_MVT kernel reference implementation:
///
/// for (int i = 0; i < N; i++) {
///   for (int j = 0; j < N; j++) {
///     x1[i] += A[i][j] * y1[j];
///   }
/// }
/// for (int i = 0; i < N; i++) {
///   for (int j = 0; j < N; j++) {
///     x2[i] += A[j][i] * y2[i];
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_MVT_HPP
#define RAJAPerf_POLYBENCH_MVT_HPP

#define POLYBENCH_MVT_BODY1 \
  x1[i] += A[j + i*N] * y1[j];

#define POLYBENCH_MVT_BODY2 \
  x2[i] += A[i + j*N] * y2[i];


#define POLYBENCH_MVT_BODY1_RAJA \
  x1view(i) += Aview(i, j) * y1view(j); 

#define POLYBENCH_MVT_BODY2_RAJA \
  x2view(i) += Aview(j, i) * y2view(i); 

#define POLYBENCH_MVT_VIEWS_RAJA \
  using VIEW_1 = RAJA::View<Real_type, \
                            RAJA::Layout<1, Index_type, 0>>; \
\
  using VIEW_2 = RAJA::View<Real_type, \
                            RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_1 x1view(x1, RAJA::Layout<1>(N)); \
  VIEW_1 x2view(x2, RAJA::Layout<1>(N)); \
  VIEW_1 y1view(y1, RAJA::Layout<1>(N)); \
  VIEW_1 y2view(y2, RAJA::Layout<1>(N)); \
  VIEW_2 Aview(A, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_MVT : public KernelBase
{
public:

  POLYBENCH_MVT(const RunParams& params);

  ~POLYBENCH_MVT();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_N;
  Real_ptr m_x1;
  Real_ptr m_x2;
  Real_ptr m_y1;
  Real_ptr m_y2;
  Real_ptr m_A;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
