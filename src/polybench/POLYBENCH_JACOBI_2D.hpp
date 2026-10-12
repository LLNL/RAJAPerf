//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_JACOBI_2D kernel reference implementation:
///
/// for (i = 1; i < N - 1; i++) {
///   for (j = 1; j < N - 1; j++) {
///     B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
///   }
/// }
/// for (i = 1; i < N - 1; i++) {
///   for (j = 1; j < N - 1; j++) {
///     A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j]);
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_JACOBI_2D_HPP
#define RAJAPerf_POLYBENCH_JACOBI_2D_HPP

#define POLYBENCH_JACOBI_2D_DATA_SETUP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  const Index_type N = m_N;


#define POLYBENCH_JACOBI_2D_BODY1 \
  B[j + i*N] = 0.2 * (A[j + i*N] + A[j-1 + i*N] + A[j+1 + i*N] + A[j + (i+1)*N] + A[j + (i-1)*N]);

#define POLYBENCH_JACOBI_2D_BODY2 \
  A[j + i*N] = 0.2 * (B[j + i*N] + B[j-1 + i*N] + B[j+1 + i*N] + B[j + (i+1)*N] + B[j + (i-1)*N]);


#define POLYBENCH_JACOBI_2D_BODY1_RAJA \
  Bview(i,j) = 0.2 * (Aview(i,j) + Aview(i,j-1) + Aview(i,j+1) + Aview(i+1,j) + Aview(i-1,j));

#define POLYBENCH_JACOBI_2D_BODY2_RAJA \
  Aview(i,j) = 0.2 * (Bview(i,j) + Bview(i,j-1) + Bview(i,j+1) + Bview(i+1,j) + Bview(i-1,j));


#define POLYBENCH_JACOBI_2D_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(N, N)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_JACOBI_2D : public KernelBase
{
public:

  POLYBENCH_JACOBI_2D(const RunParams& params);

  ~POLYBENCH_JACOBI_2D();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineKokkosVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runKokkosVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_N;

  Real_ptr m_A;
  Real_ptr m_B;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
