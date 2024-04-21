//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_HEAT_3D kernel reference implementation:
///
/// for (t = 0; t < TSTEPS; t++)
/// {
///
///   for (i = 1; i < N-1; i++) {
///     for (j = 1; j < N-1; j++) {
///       for (k = 1; k < N-1; k++) {
///         B[i][j][k] = 0.125*(A[i+1][j][k] - 2.0*A[i][j][k] + A[i-1][j][k]) +
///                      0.125*(A[i][j+1][k] - 2.0*A[i][j][k] + A[i][j-1][k]) +
///                      0.125*(A[i][j][k+1] - 2.0*A[i][j][k] + A[i][j][k-1]) +
///                      A[i][j][k];
///       }
///     }
///   }
///
///   for (i = 1; i < N-1; i++) {
///     for (j = 1; j < N-1; j++) {
///       for (k = 1; k < N-1; k++) {
///         A[i][j][k] = 0.125*(B[i+1][j][k] - 2.0*B[i][j][k] + B[i-1][j][k]) +
///                      0.125*(B[i][j+1][k] - 2.0*B[i][j][k] + B[i][j-1][k]) +
///                      0.125*(B[i][j][k+1] - 2.0*B[i][j][k] + B[i][j][k-1]) +
///                      B[i][j][k];
///       }
///     }
///   }
///
/// }


#ifndef RAJAPerf_POLYBENCH_HEAT_3D_HPP
#define RAJAPerf_POLYBENCH_HEAT_3D_HPP

#define POLYBENCH_HEAT_3D_DATA_SETUP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  \
  copyData(getDataSpace(vid), A, getDataSpace(vid), m_Ainit, m_N*m_N*m_N); \
  copyData(getDataSpace(vid), B, getDataSpace(vid), m_Binit, m_N*m_N*m_N); \
  \
  const Index_type N = m_N; \
  const Index_type tsteps = m_tsteps;


#define POLYBENCH_HEAT_3D_BODY1 \
  B[k + N*(j + N*i)] = \
                   0.125*( A[k + N*(j + N*(i+1))] - 2.0*A[k + N*(j + N*i)] + \
                           A[k + N*(j + N*(i-1))] ) + \
                   0.125*( A[k + N*(j+1 + N*i)]   - 2.0*A[k + N*(j + N*i)] + \
                           A[k + N*(j-1 + N*i)] ) + \
                   0.125*( A[k+1 + N*(j + N*i)]   - 2.0*A[k + N*(j + N*i)] + \
                           A[k-1 + N*(j + N*i)] ) + \
                   A[k + N*(j + N*i)];

#define POLYBENCH_HEAT_3D_BODY2 \
  A[k + N*(j + N*i)] = \
                   0.125*( B[k + N*(j + N*(i+1))] - 2.0*B[k + N*(j + N*i)] + \
                           B[k + N*(j + N*(i-1))] ) + \
                   0.125*( B[k + N*(j+1 + N*i)]   - 2.0*B[k + N*(j + N*i)] + \
                           B[k + N*(j-1 + N*i)] ) + \
                   0.125*( B[k+1 + N*(j + N*i)]   - 2.0*B[k + N*(j + N*i)] + \
                           B[k-1 + N*(j + N*i)] ) + \
                   B[k + N*(j + N*i)];


#define POLYBENCH_HEAT_3D_BODY1_RAJA \
  Bview(i,j,k) = \
             0.125*( Aview(i+1,j,k) - 2.0*Aview(i,j,k) + Aview(i-1,j,k) ) + \
             0.125*( Aview(i,j+1,k) - 2.0*Aview(i,j,k) + Aview(i,j-1,k) ) + \
             0.125*( Aview(i,j,k+1) - 2.0*Aview(i,j,k) + Aview(i,j,k-1) ) + \
             Aview(i,j,k);

#define POLYBENCH_HEAT_3D_BODY2_RAJA \
  Aview(i,j,k) = \
             0.125*( Bview(i+1,j,k) - 2.0*Bview(i,j,k) + Bview(i-1,j,k) ) + \
             0.125*( Bview(i,j+1,k) - 2.0*Bview(i,j,k) + Bview(i,j-1,k) ) + \
             0.125*( Bview(i,j,k+1) - 2.0*Bview(i,j,k) + Bview(i,j,k-1) ) + \
             Bview(i,j,k);


#define POLYBENCH_HEAT_3D_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<3, Index_type, 2>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<3>(N, N, N)); \
  VIEW_TYPE Bview(B, RAJA::Layout<3>(N, N, N));


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_HEAT_3D : public KernelBase
{
public:

  POLYBENCH_HEAT_3D(const RunParams& params);

  ~POLYBENCH_HEAT_3D();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_N;
  Index_type m_tsteps;

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_Ainit;
  Real_ptr m_Binit;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
