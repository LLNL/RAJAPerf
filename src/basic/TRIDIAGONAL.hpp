//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Tri-Diagonal matrix solver
/// reference implementation:
///
///      Real_ptr d = new Real_type[N];
///
///      for (Index_type i = 0; i < Nelements; ++i) {
///
///        Real_ptr Aa = Aa_global + i * N;
///        Real_ptr Ab = Ab_global + i * N;
///        Real_ptr Ac = Ac_global + i * N;
///        Real_ptr x = x_global + i * N;
///        Real_ptr b = b_global + i * N;
///
///        d[0] = Ac[0] / Ab[0];
///        x[0] =  b[0] / Ab[0];
///        for (Index_type n = 1; n < N; ++n) {
///          Real_type div = Ab[n] - Aa[n] * d[n-1];
///          d[n] = Ac[n] / div;
///          x[n] = (b[n] - Aa[n] * x[n-1]) / div;
///        }
///
///        for (Index_type n = N-2; n >= 0; --n) {
///          x[n] = x[n] - d[n] * x[n+1];
///        }
///
///      }
///
///      delete[] d;
///

#ifndef RAJAPerf_Basic_TRIDIAGONAL_HPP
#define RAJAPerf_Basic_TRIDIAGONAL_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

#define TRIDIAGONAL_DATA_SETUP                                                 \
  Real_ptr Aa_global = m_Aa_global;                                            \
  Real_ptr Ab_global = m_Ab_global;                                            \
  Real_ptr Ac_global = m_Ac_global;                                            \
  Real_ptr x_global = m_x_global;                                              \
  Real_ptr b_global = m_b_global;                                              \
  Index_type N = m_N;

#define TRIDIAGONAL_TEMP_DATA_SETUP                                            \
  Real_ptr d = new Real_type[N];

#define TRIDIAGONAL_TEMP_DATA_TEARDOWN                                         \
  delete[] d; d = nullptr;

#define TRIDIAGONAL_OFFSET(i)                                                  \
  (i)

#define TRIDIAGONAL_INDEX(n)                                                   \
  ((n) * iend)

#define TRIDIAGONAL_INDEX_TEMP(n)                                              \
  (n)

#define TRIDIAGONAL_LOCAL_DATA_SETUP                                           \
  Real_ptr Aa = Aa_global + TRIDIAGONAL_OFFSET(i);                             \
  Real_ptr Ab = Ab_global + TRIDIAGONAL_OFFSET(i);                             \
  Real_ptr Ac = Ac_global + TRIDIAGONAL_OFFSET(i);                             \
  Real_ptr x = x_global + TRIDIAGONAL_OFFSET(i);                               \
  Real_ptr b = b_global + TRIDIAGONAL_OFFSET(i);

#define TRIDIAGONAL_BODY_FORWARD                                               \
  {                                                                            \
    Index_type idx_0 = TRIDIAGONAL_INDEX(0);                                   \
    Index_type tmp_0 = TRIDIAGONAL_INDEX_TEMP(0);                              \
    d[tmp_0] = Ac[idx_0] / Ab[idx_0];                                          \
    x[idx_0] =  b[idx_0] / Ab[idx_0];                                          \
    for (Index_type n = 1; n < N; ++n) {                                       \
      Index_type idx_n = TRIDIAGONAL_INDEX(n);                                 \
      Index_type idx_m = TRIDIAGONAL_INDEX(n-1);                               \
      Index_type tmp_n = TRIDIAGONAL_INDEX_TEMP(n);                            \
      Index_type tmp_m = TRIDIAGONAL_INDEX_TEMP(n-1);                          \
      Real_type div = Ab[idx_n] - Aa[idx_n] * d[tmp_m];                        \
      d[tmp_n] = Ac[idx_n] / div;                                              \
      x[idx_n] = (b[idx_n] - Aa[idx_n] * x[idx_m]) / div;                      \
    }                                                                          \
  }

#define TRIDIAGONAL_BODY_BACKWARD                                              \
  for (Index_type n = N-2; n >= 0; --n) {                                      \
    Index_type idx_n = TRIDIAGONAL_INDEX(n);                                   \
    Index_type idx_p = TRIDIAGONAL_INDEX(n+1);                                 \
    Index_type tmp_n = TRIDIAGONAL_INDEX_TEMP(n);                              \
    x[idx_n] = x[idx_n] - d[tmp_n] * x[idx_p];                                 \
  }

#define TRIDIAGONAL_BODY_FORWARD_V2                                            \
  {                                                                            \
    Index_type idx_0 = TRIDIAGONAL_INDEX(0);                                   \
    d[idx_0] = Ac[idx_0] / Ab[idx_0];                                          \
    x[idx_0] =  b[idx_0] / Ab[idx_0];                                          \
    for (Index_type n = 1; n < N; ++n) {                                       \
      Index_type idx_n = TRIDIAGONAL_INDEX(n);                                 \
      Index_type idx_m = TRIDIAGONAL_INDEX(n-1);                               \
      Real_type div = Ab[idx_n] - Aa[idx_n] * d[idx_m];                        \
      d[idx_n] = Ac[idx_n] / div;                                              \
      x[idx_n] = (b[idx_n] - Aa[idx_n] * x[idx_m]) / div;                      \
    }                                                                          \
  }

#define TRIDIAGONAL_BODY_BACKWARD_V2                                           \
  for (Index_type n = N-2; n >= 0; --n) {                                      \
    Index_type idx_n = TRIDIAGONAL_INDEX(n);                                   \
    Index_type idx_p = TRIDIAGONAL_INDEX(n+1);                                 \
    x[idx_n] = x[idx_n] - d[idx_n] * x[idx_p];                                 \
  }

namespace rajaperf {
class RunParams;

namespace basic {

class TRIDIAGONAL : public KernelBase {
public:
  TRIDIAGONAL(const RunParams &params);

  ~TRIDIAGONAL();

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
