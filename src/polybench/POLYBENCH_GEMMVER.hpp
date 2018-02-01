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
/// POLYBENCH_GEMMVER kernel reference implementation:
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
///   }
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     x[i] = x[i] + beta * A[j][i] * y[j];
///   }
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   x[i] = x[i] + z[i];
/// }
///
/// for (Index_type i = 0; i < _PB_N; i++) {
///   for (Index_type j = 0; j < _PB_N; j++) {
///     w[i] = w[i] +  alpha * A[i][j] * x[j];
///   }
/// }
///



#ifndef RAJAPerf_POLYBENCH_GEMMVER_HXX
#define RAJAPerf_POLYBENCH_GEMMVER_HXX

#define POLYBENCH_GEMMVER_BODY1 \
  *(A + i * n + j) = *(A + i * n +j)  + *(u1 + i) * *(v1 + j) + *(u2 + i) * *(v2 + j)

#define POLYBENCH_GEMMVER_BODY2 \
  *(x + i) = *(x+i) + beta * *(A + j * n + i) * *(y + j);

#define POLYBENCH_GEMMVER_BODY3 \
  *(x + i) = *(x + i) + *(z + i);

#define POLYBENCH_GEMMVER_BODY4 \
  *(w + i) = *(w+i) + alpha * *(A + i * n + j) * *(x + j);


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
