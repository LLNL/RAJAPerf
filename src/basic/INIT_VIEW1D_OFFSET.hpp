//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT_VIEW1D_OFFSET kernel reference implementation:
///
/// const Real_type val = ...;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i-ibegin] = i * val;
/// }
///
/// RAJA variants use a "View" and "Layout" to do the same thing. These
/// RAJA constructs provide little benfit in 1D, but they are used here
/// to exercise those RAJA mechanics in the simplest scenario.
///

#ifndef RAJAPerf_Basic_INIT_VIEW1D_OFFSET_HPP
#define RAJAPerf_Basic_INIT_VIEW1D_OFFSET_HPP


#define INIT_VIEW1D_OFFSET_DATA_SETUP \
  Real_ptr a = m_a; \
  const Real_type v = m_val;

#define INIT_VIEW1D_OFFSET_BODY  \
    a[i-ibegin] = i * v;

#define INIT_VIEW1D_OFFSET_BODY_RAJA  \
    view(i) = i * v;

#define INIT_VIEW1D_OFFSET_VIEW_RAJA  \
  using ViewType = RAJA::View<Real_type, RAJA::OffsetLayout<1> >; \
  ViewType view(a, RAJA::make_offset_layout<1>({{1}}, {{iend+1}}));


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class INIT_VIEW1D_OFFSET : public KernelBase
{
public:

  INIT_VIEW1D_OFFSET(const RunParams& params);

  ~INIT_VIEW1D_OFFSET();

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
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_a;
  Real_type m_val;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
