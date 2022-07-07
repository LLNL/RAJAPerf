//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SORT kernel reference implementation:
///
/// std::sort(x+ibegin, x+iend);
///

#ifndef RAJAPerf_Algorithm_SORT_HPP
#define RAJAPerf_Algorithm_SORT_HPP

#define SORT_DATA_SETUP \
  Real_ptr x = m_x;

#define STD_SORT_ARGS  \
  x + iend*irep + ibegin, x + iend*irep + iend

#define RAJA_SORT_ARGS \
  RAJA::make_span(x + iend*irep + ibegin, iend - ibegin)


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class SORT : public KernelBase
{
public:

  SORT(const RunParams& params);

  ~SORT();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
  {
    getCout() << "\n  SORT : Unknown OMP Target variant id = " << vid << std::endl;
  }
  void runStdParVariant(VariantID vid, size_t tune_idx);

private:
  static const size_t default_gpu_block_size = 0;

  Real_ptr m_x;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
