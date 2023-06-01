//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D mass matrix via partial assembly
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation
/// https://github.com/mfem/mfem/blob/master/fem/bilininteg_mass_ea.cpp#L142
///
/// for (int e = 0; e < NE; ++e) {
///
///
/// } // element loop
///

#ifndef RAJAPerf_Apps_MASS3DEA_HPP
#define RAJAPerf_Apps_MASS3DEA_HPP

#define MASS3DEA_DATA_SETUP \
Real_ptr B = m_B; \
Real_ptr Bt = m_Bt; \
Real_ptr D = m_D; \
Real_ptr M = m_M; \
Index_type NE = m_NE;

#include "common/KernelBase.hpp"
#include "FEM_MACROS.hpp"

#include "RAJA/RAJA.hpp"

//Number of Dofs/Qpts in 1D
#define MEA_D1D 4
#define MEA_Q1D 5
#define B_(x, y) B[x + MEA_Q1D * y]
#define Bt_(x, y) Bt[x + MEA_D1D * y]
#define M_(i1, i2, i3, j1, j2, j3, e)                                      \
  M[i1 + MEA_D1D * (i2 + MEA_D1D * (i3 + MEA_D1D * (j1 + MEA_D1D * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))))]

#define D_(qx, qy, qz, e)                                                      \
  D[qx + MEA_Q1D * qy + MEA_Q1D * MEA_Q1D * qz + MEA_Q1D * MEA_Q1D * MEA_Q1D * e]




namespace rajaperf
{
class RunParams;

namespace apps
{

class MASS3DEA : public KernelBase
{
public:

  MASS3DEA(const RunParams& params);

  ~MASS3DEA();

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
  static const size_t default_gpu_block_size = MEA_Q1D * MEA_Q1D;
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_D;
  Real_ptr m_M;

  Index_type m_NE;
  Index_type m_NE_default;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard