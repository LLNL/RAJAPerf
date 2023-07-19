//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Assembly of 3D mass matrix
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation
/// https://github.com/mfem/mfem/blob/master/fem/integ/bilininteg_mass_ea.cpp#L142
///
/// for (int e = 0; e < NE; ++e)
///   {
///
///     double s_B[MQ1s][MD1s];
///     double r_B[MQ1r][MD1r];
///
///     double (*l_B)[MD1] = nullptr;
///
///     for(int d=0; d<D1D; ++d) {
///       for(int q=0; q<Q1D; ++q) {
///         s_B[q][d] = B(q,d);
///       }
///     }
///
///     l_B = (double (*)[MD1])s_B;
///
///     double s_D[MQ1][MQ1][MQ1];
///
///     for(int k1=0; k1<Q1D; ++k1) {
///       for(int k2=0; k2<Q1D; ++k2) {
///         for(int k3=0; k3<Q1D; ++k3) {
///           s_D[k1][k2][k3] = D(k1,k2,k3,e);
///         }
///       }
///     }
///
///     for(int i1=0; i1<D1D; ++i1) {
///       for(int i2=0; i2<D1D; ++i2) {
///         for(int i3=0; i3<D1D; ++i3) {
///
///           for (int j1 = 0; j1 < D1D; ++j1) {
///             for (int j2 = 0; j2 < D1D; ++j2) {
///               for (int j3 = 0; j3 < D1D; ++j3) {
///
///                 double val = 0.0;
///                 for (int k1 = 0; k1 < Q1D; ++k1) {
///                   for (int k2 = 0; k2 < Q1D; ++k2) {
///                     for (int k3 = 0; k3 < Q1D; ++k3) {
///
///                       val += l_B[k1][i1] * l_B[k1][j1]
///                         * l_B[k2][i2] * l_B[k2][j2]
///                         * l_B[k3][i3] * l_B[k3][j3]
///                         * s_D[k1][k2][k3];
///                     }
///                   }
///                 }
///
///                 M(i1, i2, i3, j1, j2, j3, e) = val;
///               }
///             }
///           }
///
///         }
///       }
///     }
///
///   } // element loop
///

#ifndef RAJAPerf_Apps_MASS3DEA_HPP
#define RAJAPerf_Apps_MASS3DEA_HPP

#define MASS3DEA_DATA_SETUP                                             \
  Real_ptr B = m_B;                                                     \
  Real_ptr D = m_D;                                                     \
  Real_ptr M = m_M;                                                     \
  Index_type NE = m_NE;

#include "common/KernelBase.hpp"
#include "FEM_MACROS.hpp"

#include "RAJA/RAJA.hpp"

// Number of Dofs/Qpts in 1D
#define MEA_D1D 4
#define MEA_Q1D 5
#define B_MEA_(x, y) B[x + MEA_Q1D * y]
#define M_(i1, i2, i3, j1, j2, j3, e)                                   \
  M[i1 + MEA_D1D * (i2 + MEA_D1D * (i3 + MEA_D1D * (j1 + MEA_D1D * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))))]

#define D_MEA_(qx, qy, qz, e)                                           \
  D[qx + MEA_Q1D * qy + MEA_Q1D * MEA_Q1D * qz +                        \
    MEA_Q1D * MEA_Q1D * MEA_Q1D * e]

#define MASS3DEA_0 RAJA_TEAM_SHARED double s_B[MEA_Q1D][MEA_D1D];

#define MASS3DEA_0_CPU double s_B[MEA_Q1D][MEA_D1D];

#define MASS3DEA_1 s_B[q][d] = B_MEA_(q, d);

#define MASS3DEA_2                                                      \
  double(*l_B)[MEA_D1D] = (double(*)[MEA_D1D])s_B;                      \
  RAJA_TEAM_SHARED double s_D[MEA_Q1D][MEA_Q1D][MEA_Q1D];

#define MASS3DEA_2_CPU                                                  \
  double(*l_B)[MEA_D1D] = (double(*)[MEA_D1D])s_B;                      \
  double s_D[MEA_Q1D][MEA_Q1D][MEA_Q1D];

#define MASS3DEA_3 s_D[k1][k2][k3] = D_MEA_(k1, k2, k3, e);

#define MASS3DEA_4                                                      \
  for (int j1 = 0; j1 < MEA_D1D; ++j1) {                                \
    for (int j2 = 0; j2 < MEA_D1D; ++j2) {                              \
      for (int j3 = 0; j3 < MEA_D1D; ++j3) {                            \
                                                                        \
        double val = 0.0;                                               \
        for (int k1 = 0; k1 < MEA_Q1D; ++k1) {                          \
          for (int k2 = 0; k2 < MEA_Q1D; ++k2) {                        \
            for (int k3 = 0; k3 < MEA_Q1D; ++k3) {                      \
                                                                        \
              val += l_B[k1][i1] * l_B[k1][j1] * l_B[k2][i2]            \
                * l_B[k2][j2] *                                         \
                l_B[k3][i3] * l_B[k3][j3] * s_D[k1][k2][k3];            \
            }                                                           \
          }                                                             \
        }                                                               \
        M_(i1, i2, i3, j1, j2, j3, e) = val;                            \
      }                                                                 \
    }                                                                   \
  }

namespace rajaperf {
class RunParams;

namespace apps {

class MASS3DEA : public KernelBase {
public:
  MASS3DEA(const RunParams &params);

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
  template <size_t block_size> void runCudaVariantImpl(VariantID vid);
  template <size_t block_size> void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = MEA_D1D * MEA_D1D * MEA_D1D;
  using gpu_block_sizes_type =
      gpu_block_size::list_type<default_gpu_block_size>;

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
