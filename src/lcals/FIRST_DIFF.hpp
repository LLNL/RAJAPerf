//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FIRST_DIFF kernel reference implementation:
///
/// for (Index_type i = 0; i < N-1; ++i ) {
///   x[i] = y[i+1] - y[i];
/// }
///

#ifndef RAJAPerf_Lcals_FIRST_DIFF_HPP
#define RAJAPerf_Lcals_FIRST_DIFF_HPP


#define FIRST_DIFF_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y;

#define FIRST_DIFF_BODY  \
  x[i] = y[i+1] - y[i];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class FIRST_DIFF : public KernelBase
{
public:

  FIRST_DIFF(const RunParams& params);

  ~FIRST_DIFF();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  bool isGPUBlockSizeSupported() const;
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_ptr m_y;

  Index_type m_N;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
