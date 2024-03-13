//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT3 kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;
/// }
///

#ifndef RAJAPerf_Basic_INIT3_HPP
#define RAJAPerf_Basic_INIT3_HPP


#define INIT3_DATA_SETUP \
  Real_ptr out1 = m_out1; \
  Real_ptr out2 = m_out2; \
  Real_ptr out3 = m_out3; \
  Real_ptr in1 = m_in1; \
  Real_ptr in2 = m_in2;

#define INIT3_BODY  \
  out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class INIT3 : public KernelBase
{
public:

  INIT3(const RunParams& params);

  ~INIT3();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runKokkosVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t block_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_out1;
  Real_ptr m_out2;
  Real_ptr m_out3;
  Real_ptr m_in1;
  Real_ptr m_in2;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
