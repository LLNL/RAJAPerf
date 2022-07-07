//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// GEN_LIN_RECUR kernel reference implementation:
///
/// Note: kernel is altered to enable parallelism and reproducibility
///       (in original, stb5 is a scalar). In the future, this may be
///       changed to use atomics. --RDH
///
/// Index_type kb5i = 0;
///
/// for (Index_type k = 0; k < N; ++k ) {
///   b5[k+kb5i] = sa[k] + stb5[k]*sb[k];
///   stb5[k] = b5[k+kb5i] - stb5[k];
/// }
///
/// for (Index_type i = 1; i < N+1; ++i ) {
///   Index_type k = N - i ;
///   b5[k+kb5i] = sa[k] + stb5[k]*sb[k];
///   stb5[k] = b5[k+kb5i] - stb5[k];
/// }
///

#ifndef RAJAPerf_Lcals_GEN_LIN_RECUR_HPP
#define RAJAPerf_Lcals_GEN_LIN_RECUR_HPP


#define GEN_LIN_RECUR_DATA_SETUP \
  Real_ptr b5 = m_b5; \
  Real_ptr sa = m_sa; \
  Real_ptr sb = m_sb; \
  Real_ptr stb5 = m_stb5; \
\
  Index_type kb5i = m_kb5i; \
  Index_type N = m_N;

#define GEN_LIN_RECUR_BODY1  \
  b5[k+kb5i] = sa[k] + stb5[k]*sb[k]; \
  stb5[k] = b5[k+kb5i] - stb5[k];

#define GEN_LIN_RECUR_BODY2  \
  Index_type k = N - i ; \
  b5[k+kb5i] = sa[k] + stb5[k]*sb[k]; \
  stb5[k] = b5[k+kb5i] - stb5[k];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class GEN_LIN_RECUR : public KernelBase
{
public:

  GEN_LIN_RECUR(const RunParams& params);

  ~GEN_LIN_RECUR();

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
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_b5;
  Real_ptr m_sa;
  Real_ptr m_sb;
  Real_ptr m_stb5;
  Index_type m_kb5i;

  Index_type m_N;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
