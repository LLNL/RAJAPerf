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

#define TRIDIAGONAL_LOCAL_DATA_SETUP                                           \
  Real_ptr Aa = Aa_global + i * N;                                             \
  Real_ptr Ab = Ab_global + i * N;                                             \
  Real_ptr Ac = Ac_global + i * N;                                             \
  Real_ptr x = x_global + i * N;                                               \
  Real_ptr b = b_global + i * N;

#define TRIDIAGONAL_BODY_FORWARD                                               \
  d[0] = Ac[0] / Ab[0];                                                        \
  x[0] =  b[0] / Ab[0];                                                        \
  for (Index_type n = 1; n < N; ++n) {                                         \
    Real_type div = Ab[n] - Aa[n] * d[n-1];                                    \
    d[n] = Ac[n] / div;                                                        \
    x[n] = (b[n] - Aa[n] * x[n-1]) / div;                                      \
  }

#define TRIDIAGONAL_BODY_BACKWARD                                              \
  for (Index_type n = N-2; n >= 0; --n) {                                      \
    x[n] = x[n] - d[n] * x[n+1];                                               \
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
