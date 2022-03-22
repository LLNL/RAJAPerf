//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Tri-Diagonal parallel matrix solver
/// reference implementation:
///
///      Real_type E_[N-1]; // lower diagonal of A
///      Real_ptr E = E_ - 2; // [2:N]
///      for (int j = 2; j <= N; ++j) { // par
///        E[j] = Aa[j-1];
///      }
///      Real_type F_[N-1]; // upper diagonal of A
///      Real_ptr F = F_ - 1; // [1:N-1]
///      for (int j = 1; j <= N-1; ++j) { // par
///        F[j] = Ac[j-1];
///      }
///      Real_type D_[N]; // diagonal of A
///      Real_ptr D = D_ - 1; // [1:N]
///      for (int j = 1; j <= N; ++j) { // par
///        D[j] = Ab[j-1];
///      }
///      Real_type B_[N]; // rhs of equation
///      Real_ptr B = B_ - 1; // [1:N]
///      for (int j = 1; j <= N; ++j) { // par
///        B[j] = b[j-1];
///      }
///
///      Real_type EF_[N]; // holds products (-e[i]*f[i-1])
///      Real_ptr EF = EF_ - 1; // [1:N]
///      Real_type TEMP_[N]; // temporary array
///      Real_ptr TEMP = TEMP_ - 1; // [1:N]
///      Real_type QI_[N]; // Qi[j]
///      Real_ptr QI = QI_ - 1; // [1:N]
///      Real_type QIM1_[N+1]; // Qi-1[j]
///      Real_ptr QIM1 = QIM1_ - 0; // [0:N]
///      Real_type QIM2_[N+2]; // Qi-1[j]
///      Real_ptr QIM2 = QIM2_ - (-1); // [-1:N]
///
///      Real_type U_[N];
///      Real_ptr U = U_ - 1; // [1:N]
///
///      // Real_type M_[N-1];
///      // Real_ptr M = M_ - 2; // [2:N]
///      Real_type M_[N];
///      Real_ptr M = M_ - 1; // [1:N]
///
///      Real_type Y_[N];
///      Real_ptr Y = Y_ - 1; // [1:N]
///
///      Real_type X_[N];
///      Real_ptr X = X_ - 1; // [1:N]
///
///      EF[1] = 0;
///      for (int j = 2; j <= N; ++j) { // par
///        EF[j] = -E[j] * F[j-1];
///      }
///      for (int j = -1; j <= N; ++j) { // par
///        QIM2[j] = 1;
///      }
///      QIM1[0] = 1;
///      for (int j = 1; j <= N; ++j) { // par
///        QIM1[j] = D[j];
///      }
///      QI[1] = D[1];
///      for (int j = 2; j <= N; ++j) { // par
///        QI[j] = D[j] * D[j-1] + EF[j];
///      }
///      for (int i = 2; i <= N; i *= 2) {
///        for (int j = i-1; j <= N; ++j) { // par
///          TEMP[j] = QIM1[j] * QIM1[j-i+1] + EF[j-i+2] * QIM2[j] * QIM2[j-i];
///        }
///        for (int j = N; j >= i; --j) { // par (beware)
///          QIM1[j] = QI[j] * QIM1[j-i] + EF[j-i+1] * QIM1[j] * QIM2[j-i-1];
///        }
///        for (int j = i-1; j <= N; ++j) { // par
///          QIM2[j] = TEMP[j];
///        }
///        for (int j = i+1; j <= N; ++j) { // par
///          QI[j] = D[j] * QIM1[j-1] + EF[j] * QIM2[j-2];
///        }
///      }
///
///      U[1] = QI[1];
///      for (int j = 2; j <= N; ++j) { // par
///        U[j] = QI[j] / QI[j-1];
///      }
///      for (int j = 2; j <= N; ++j) { // par
///        M[j] = E[j] / U[j-1];
///      }
///      for (int j = 1; j <= N; ++j) { // par
///        Y[j] = B[j];
///      }
///      M[1] = 0;
///      for (int j = 2; j <= N; ++j) { // par
///        M[j] = -M[j];
///      }
///
///      for (int i = 1; i <= N; i *= 2) {
///        for (int j = N; j >= i+1; --j) { // par (beware)
///          Y[j] = Y[j] + Y[j-i] * M[j];
///        }
///        for (int j = N; j >= i+1; --j) { // par (beware)
///          M[j] = M[j] * M[j-i];
///        }
///      }
///
///      for (int j = 1; j <= N; ++j) { // par
///        X[j] = Y[j] / U[j];
///      }
///      for (int j = 1; j <= N-1; ++j) { // par
///        M[j] = -F[j] / U[j];
///      }
///      M[N] = 0;
///      for (int i = 1; i <= N; i *= 2) {
///        for (int j = 1; j <= N-i; ++j) { // par (beware)
///          X[j] = X[j] + X[j+i] * M[j];
///        }
///        for (int j = 1; j <= N-i; ++j) { // par (beware)
///          M[j] = M[j] * M[j+i];
///        }
///      }
///
///      for (int i = 1; i <= N; ++i) { // par
///        x[i-1] = X[i];
///      }
///

#ifndef RAJAPerf_Basic_TRIDIAGONAL_PAR_HPP
#define RAJAPerf_Basic_TRIDIAGONAL_PAR_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

#define TRIDIAGONAL_PAR_DATA_SETUP                                             \
  Real_ptr Aa_global = m_Aa_global;                                            \
  Real_ptr Ab_global = m_Ab_global;                                            \
  Real_ptr Ac_global = m_Ac_global;                                            \
  Real_ptr x_global = m_x_global;                                              \
  Real_ptr b_global = m_b_global;                                              \
  Index_type N = m_N;

#define TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL                                  \
  Real_ptr d = new Real_type[N];

#define TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL                               \
  delete[] d; d = nullptr;

#define TRIDIAGONAL_PAR_OFFSET(i)                                              \
  (i)

#define TRIDIAGONAL_PAR_INDEX(n)                                               \
  ((n) * iend)

#define TRIDIAGONAL_PAR_INDEX_LOCAL(n)                                         \
  (n)

#define TRIDIAGONAL_PAR_LOCAL_DATA_SETUP                                       \
  Real_ptr Aa = Aa_global + TRIDIAGONAL_PAR_OFFSET(i);                         \
  Real_ptr Ab = Ab_global + TRIDIAGONAL_PAR_OFFSET(i);                         \
  Real_ptr Ac = Ac_global + TRIDIAGONAL_PAR_OFFSET(i);                         \
  Real_ptr x = x_global + TRIDIAGONAL_PAR_OFFSET(i);                           \
  Real_ptr b = b_global + TRIDIAGONAL_PAR_OFFSET(i);

#define TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL                                \
  {                                                                            \
    Index_type idx_0 = TRIDIAGONAL_PAR_INDEX(0);                               \
    Index_type tmp_0 = TRIDIAGONAL_PAR_INDEX_LOCAL(0);                         \
    d[tmp_0] = Ac[idx_0] / Ab[idx_0];                                          \
    x[idx_0] =  b[idx_0] / Ab[idx_0];                                          \
    for (Index_type n = 1; n < N; ++n) {                                       \
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);                             \
      Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);                           \
      Index_type tmp_n = TRIDIAGONAL_PAR_INDEX_LOCAL(n);                       \
      Index_type tmp_m = TRIDIAGONAL_PAR_INDEX_LOCAL(n-1);                     \
      Real_type div = Ab[idx_n] - Aa[idx_n] * d[tmp_m];                        \
      d[tmp_n] = Ac[idx_n] / div;                                              \
      x[idx_n] = (b[idx_n] - Aa[idx_n] * x[idx_m]) / div;                      \
    }                                                                          \
  }

#define TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL                               \
  for (Index_type n = N-2; n >= 0; --n) {                                      \
    Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);                               \
    Index_type idx_p = TRIDIAGONAL_PAR_INDEX(n+1);                             \
    Index_type tmp_n = TRIDIAGONAL_PAR_INDEX_LOCAL(n);                         \
    x[idx_n] = x[idx_n] - d[tmp_n] * x[idx_p];                                 \
  }

#define TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_GLOBAL                               \
  {                                                                            \
    Index_type idx_0 = TRIDIAGONAL_PAR_INDEX(0);                               \
    d[idx_0] = Ac[idx_0] / Ab[idx_0];                                          \
    x[idx_0] =  b[idx_0] / Ab[idx_0];                                          \
    for (Index_type n = 1; n < N; ++n) {                                       \
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);                             \
      Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);                           \
      Real_type div = Ab[idx_n] - Aa[idx_n] * d[idx_m];                        \
      d[idx_n] = Ac[idx_n] / div;                                              \
      x[idx_n] = (b[idx_n] - Aa[idx_n] * x[idx_m]) / div;                      \
    }                                                                          \
  }

#define TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_GLOBAL                              \
  for (Index_type n = N-2; n >= 0; --n) {                                      \
    Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);                               \
    Index_type idx_p = TRIDIAGONAL_PAR_INDEX(n+1);                             \
    x[idx_n] = x[idx_n] - d[idx_n] * x[idx_p];                                 \
  }


namespace rajaperf {
class RunParams;

namespace basic {

class TRIDIAGONAL_PAR : public KernelBase {
public:
  TRIDIAGONAL_PAR(const RunParams &params);

  ~TRIDIAGONAL_PAR();

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
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

  static const Index_type N_default = 60;

  Real_ptr m_Aa_global;
  Real_ptr m_Ab_global;
  Real_ptr m_Ac_global;
  Real_ptr m_x_global;
  Real_ptr m_b_global;

  Index_type m_N;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
