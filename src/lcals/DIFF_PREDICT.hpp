//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DIFF_PREDICT kernel reference implementation:
///
/// Index_type offset = iend - ibegin;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   ar                  = cx[i + offset * 4];
///   br                  = ar - px[i + offset * 4];
///   px[i + offset * 4]  = ar;
///   cr                  = br - px[i + offset * 5];
///   px[i + offset * 5]  = br;
///   ar                  = cr - px[i + offset * 6];
///   px[i + offset * 6]  = cr;
///   br                  = ar - px[i + offset * 7];
///   px[i + offset * 7]  = ar;
///   cr                  = br - px[i + offset * 8];
///   px[i + offset * 8]  = br;
///   ar                  = cr - px[i + offset * 9];
///   px[i + offset * 9]  = cr;
///   br                  = ar - px[i + offset * 10];
///   px[i + offset * 10] = ar;
///   cr                  = br - px[i + offset * 11];
///   px[i + offset * 11] = br;
///   px[i + offset * 13] = cr - px[i + offset * 12];
///   px[i + offset * 12] = cr;
/// }
///

#ifndef RAJAPerf_Lcals_DIFF_PREDICT_HPP
#define RAJAPerf_Lcals_DIFF_PREDICT_HPP


#define DIFF_PREDICT_DATA_SETUP \
  Real_ptr px = m_px; \
  Real_ptr cx = m_cx; \
  const Index_type offset = m_offset;

#define DIFF_PREDICT_BODY  \
  Real_type ar, br, cr; \
\
  ar                  = cx[i + offset * 4];       \
  br                  = ar - px[i + offset * 4];  \
  px[i + offset * 4]  = ar;                       \
  cr                  = br - px[i + offset * 5];  \
  px[i + offset * 5]  = br;                       \
  ar                  = cr - px[i + offset * 6];  \
  px[i + offset * 6]  = cr;                       \
  br                  = ar - px[i + offset * 7];  \
  px[i + offset * 7]  = ar;                       \
  cr                  = br - px[i + offset * 8];  \
  px[i + offset * 8]  = br;                       \
  ar                  = cr - px[i + offset * 9];  \
  px[i + offset * 9]  = cr;                       \
  br                  = ar - px[i + offset * 10]; \
  px[i + offset * 10] = ar;                       \
  cr                  = br - px[i + offset * 11]; \
  px[i + offset * 11] = br;                       \
  px[i + offset * 13] = cr - px[i + offset * 12]; \
  px[i + offset * 12] = cr;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class DIFF_PREDICT : public KernelBase
{
public:

  DIFF_PREDICT(const RunParams& params);

  ~DIFF_PREDICT();

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

  Real_ptr m_px;
  Real_ptr m_cx;

  Index_type m_array_length;
  Index_type m_offset;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
