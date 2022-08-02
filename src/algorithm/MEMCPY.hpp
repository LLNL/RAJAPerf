//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// MEMCPY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   x[i] = val ;
/// }
///

#ifndef RAJAPerf_Algorithm_MEMCPY_HPP
#define RAJAPerf_Algorithm_MEMCPY_HPP

#define MEMCPY_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \

#define MEMCPY_STD_ARGS  \
  y + ibegin, x + ibegin, (iend-ibegin)*sizeof(Real_type)

#define MEMCPY_BODY \
  y[i] = x[i];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class MEMCPY : public KernelBase
{
public:

  MEMCPY(const RunParams& params);

  ~MEMCPY();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void runSeqVariantDefault(VariantID vid);
  void runSeqVariantLibrary(VariantID vid);

  template < size_t block_size >
  void runCudaVariantBlock(VariantID vid);
  void runCudaVariantLibrary(VariantID vid);

  template < size_t block_size >
  void runHipVariantBlock(VariantID vid);
  void runHipVariantLibrary(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_ptr m_y;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
