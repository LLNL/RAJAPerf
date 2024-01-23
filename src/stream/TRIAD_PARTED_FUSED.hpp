//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRIAD_PARTED_FUSED kernel reference implementation:
///
/// for (size_t p = 1; p < parts.size(); ++p ) {
///   Index_type ibegin = iparts[p-1];
///   Index_type iend = iparts[p];
///   for (Index_type i = ibegin; i < iend; ++i ) {
///     a[i] = b[i] + alpha * c[i] ;
///   }
/// }
///

#ifndef RAJAPerf_Stream_TRIAD_PARTED_FUSED_HPP
#define RAJAPerf_Stream_TRIAD_PARTED_FUSED_HPP

#define TRIAD_PARTED_FUSED_DATA_SETUP \
  std::vector<Index_type> parts = m_parts; \
  \
  Real_ptr a = m_a; \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_type alpha = m_alpha;

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_SETUP \
  triad_holder* triad_holders = new triad_holder[parts.size()-1];

#define TRIAD_PARTED_FUSED_MANUAL_FUSER_TEARDOWN \
  delete[] triad_holders;


#define TRIAD_PARTED_FUSED_BODY  \
  a[i] = b[i] + alpha * c[i] ;


#define TRIAD_PARTED_FUSED_MANUAL_LAMBDA_FUSER_SETUP \
  auto make_lambda = [](Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha, Index_type ibegin) { \
    return [=](Index_type ii) { \
      Index_type i = ii + ibegin; \
      TRIAD_PARTED_FUSED_BODY; \
    }; \
  }; \
  using lambda_type = decltype(make_lambda(Real_ptr(), Real_ptr(), Real_ptr(), Real_type(), Index_type())); \
  lambda_type* lambdas = reinterpret_cast<lambda_type*>( \
      malloc(sizeof(lambda_type) * (parts.size()-1))); \
  Index_type* lens = new Index_type[parts.size()-1];

#define TRIAD_PARTED_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN \
  free(lambdas); \
  delete[] lens;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace stream
{

struct alignas(2*alignof(void*)) triad_holder {
  Index_type len;
  Real_ptr a;
  Real_ptr b;
  Real_ptr c;
  Real_type alpha;
  Index_type ibegin;
};

class TRIAD_PARTED_FUSED : public KernelBase
{
public:

  TRIAD_PARTED_FUSED(const RunParams& params);

  ~TRIAD_PARTED_FUSED();

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
  void runCudaVariantGraphReuse(VariantID vid);
  template < size_t block_size >
  void runCudaVariantSOA2dSync(VariantID vid);
  template < size_t block_size >
  void runHipVariantSOA2dSync(VariantID vid);
  template < size_t block_size >
  void runCudaVariantSOA2dReuse(VariantID vid);
  template < size_t block_size >
  void runHipVariantSOA2dReuse(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOS2dSync(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOS2dSync(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOS2dPoolSync(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOS2dPoolSync(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOS2dReuse(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOS2dReuse(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOS2dReuseFunctionPointer(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOS2dReuseFunctionPointer(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOS2dReuseVirtualFunction(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOS2dReuseVirtualFunction(VariantID vid);
  template < size_t block_size >
  void runCudaVariantAOSScanReuse(VariantID vid);
  template < size_t block_size >
  void runHipVariantAOSScanReuse(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  std::vector<Index_type> m_parts;

  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
