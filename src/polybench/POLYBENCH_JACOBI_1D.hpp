//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_JACOBI_1D kernel reference implementation:
///
/// for (i = 1; i < N - 1; i++) {
///   B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
/// }
/// for (i = 1; i < N - 1; i++) {
///   A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
/// }


#ifndef RAJAPerf_POLYBENCH_JACOBI_1D_HPP
#define RAJAPerf_POLYBENCH_JACOBI_1D_HPP

#define POLYBENCH_JACOBI_1D_DATA_SETUP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  const Index_type N = m_N;


#define POLYBENCH_JACOBI_1D_BODY1 \
  B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);

#define POLYBENCH_JACOBI_1D_BODY2 \
  A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_JACOBI_1D : public KernelBase
{
public:

  POLYBENCH_JACOBI_1D(const RunParams& params);

  ~POLYBENCH_JACOBI_1D();

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
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Index_type m_N;

  Real_ptr m_A;
  Real_ptr m_B;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
