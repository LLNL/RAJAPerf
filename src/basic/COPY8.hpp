//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// COPY8 kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y0[i] = x0[i] ;
///   y1[i] = x1[i] ;
///   y2[i] = x2[i] ;
///   y3[i] = x3[i] ;
///   y4[i] = x4[i] ;
///   y5[i] = x5[i] ;
///   y6[i] = x6[i] ;
///   y7[i] = x7[i] ;
/// }
///

#ifndef RAJAPerf_Basic_COPY8_HPP
#define RAJAPerf_Basic_COPY8_HPP

#define COPY8_DATA_SETUP \
  Real_ptr x0 = m_x0; \
  Real_ptr x1 = m_x1; \
  Real_ptr x2 = m_x2; \
  Real_ptr x3 = m_x3; \
  Real_ptr x4 = m_x4; \
  Real_ptr x5 = m_x5; \
  Real_ptr x6 = m_x6; \
  Real_ptr x7 = m_x7; \
  Real_ptr y0 = m_y0; \
  Real_ptr y1 = m_y1; \
  Real_ptr y2 = m_y2; \
  Real_ptr y3 = m_y3; \
  Real_ptr y4 = m_y4; \
  Real_ptr y5 = m_y5; \
  Real_ptr y6 = m_y6; \
  Real_ptr y7 = m_y7;

#define COPY8_BODY  \
  y0[i] = x0[i] ; \
  y1[i] = x1[i] ; \
  y2[i] = x2[i] ; \
  y3[i] = x3[i] ; \
  y4[i] = x4[i] ; \
  y5[i] = x5[i] ; \
  y6[i] = x6[i] ; \
  y7[i] = x7[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class COPY8 : public KernelBase
{
public:

  COPY8(const RunParams& params);

  ~COPY8();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
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

  Real_ptr m_x0;
  Real_ptr m_x1;
  Real_ptr m_x2;
  Real_ptr m_x3;
  Real_ptr m_x4;
  Real_ptr m_x5;
  Real_ptr m_x6;
  Real_ptr m_x7;
  Real_ptr m_y0;
  Real_ptr m_y1;
  Real_ptr m_y2;
  Real_ptr m_y3;
  Real_ptr m_y4;
  Real_ptr m_y5;
  Real_ptr m_y6;
  Real_ptr m_y7;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
