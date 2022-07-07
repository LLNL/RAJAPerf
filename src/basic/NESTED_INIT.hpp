//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NESTED_INIT kernel reference implementation:
///
/// for (Index_type k = 0; k < nk; ++k ) {
///   for (Index_type j = 0; j < nj; ++j ) {
///     for (Index_type i = 0; i < ni; ++i ) {
///       array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
///     }
///   }
/// }
///

#ifndef RAJAPerf_Basic_NESTED_INIT_HPP
#define RAJAPerf_Basic_NESTED_INIT_HPP


#define NESTED_INIT_DATA_SETUP \
  Real_ptr array = m_array; \
  Index_type ni = m_ni; \
  Index_type nj = m_nj; \
  Index_type nk = m_nk;

#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class NESTED_INIT : public KernelBase
{
public:

  NESTED_INIT(const RunParams& params);

  ~NESTED_INIT();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);
  void runKokkosVariant(VariantID vid, size_t tune_idx);

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

  Index_type m_array_length;

  Real_ptr m_array;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_n_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
