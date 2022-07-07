//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_FLOYD_WARSHALL kernel reference implementation:
///
/// Note: kernel is altered to enable parallelism (original did not have
///       separate input and output arrays).
///
/// for (Index_type k = 0; k < N; k++) {
///   for (Index_type i = 0; i < N; i++) {
///     for (Index_type j = 0; j < N; j++) {
///       pout[i][j] = pin[i][j] < pin[i][k] + pin[k][j] ?
///                    pin[i][j] : pin[i][k] + pin[k][j];
///     }
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_FLOYD_WARSHALL_HPP
#define RAJAPerf_POLYBENCH_FLOYD_WARSHALL_HPP

#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP \
  Real_ptr pin = m_pin; \
  Real_ptr pout = m_pout; \
  const Index_type N = m_N;


#define POLYBENCH_FLOYD_WARSHALL_BODY \
  pout[j + i*N] = pin[j + i*N] < pin[k + i*N] + pin[j + k*N] ? \
                  pin[j + i*N] : pin[k + i*N] + pin[j + k*N];


#define POLYBENCH_FLOYD_WARSHALL_BODY_RAJA \
  poutview(i, j) = pinview(i, j) < pinview(i, k) + pinview(k, j) ? \
                   pinview(i, j) : pinview(i, k) + pinview(k, j);


#define POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, \
                               RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE pinview(pin, RAJA::Layout<2>(N, N)); \
  VIEW_TYPE poutview(pout, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf
{

class RunParams;

namespace polybench
{

class POLYBENCH_FLOYD_WARSHALL : public KernelBase
{
public:

  POLYBENCH_FLOYD_WARSHALL(const RunParams& params);

  ~POLYBENCH_FLOYD_WARSHALL();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size,
                                                         gpu_block_size::MultipleOf<32>>;

  Index_type m_N;

  Real_ptr m_pin;
  Real_ptr m_pout;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
