//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// PI_REDUCE kernel reference implementation:
///
/// const int N = ...;  -- num [0, 1] sub-intervals used in Riemann integration
/// const double dx = 1.0 / double(num_bins);
///
/// double pi = 0.0;
/// for (Index_type i = 0; i < N; ++i ) {
///   double x = (double(i) + 0.5) * dx;
///   pi += dx / (1.0 + x * x);
/// }
/// pi *= 4.0;
///

#ifndef RAJAPerf_Basic_PI_REDUCE_HPP
#define RAJAPerf_Basic_PI_REDUCE_HPP

#define PI_REDUCE_DATA_SETUP \
  Real_type dx = m_dx;

#define PI_REDUCE_BODY \
  double x = (double(i) + 0.5) * dx; \
  pi += dx / (1.0 + x * x);

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class PI_REDUCE : public KernelBase
{
public:

  PI_REDUCE(const RunParams& params);

  ~PI_REDUCE();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void setOpenMPTuningDefinitions(VariantID vid);
  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setOpenMPTargetTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid);

  template < size_t block_size, typename MappingHelper >
  void runCudaVariantBase(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runCudaVariantRAJA(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runCudaVariantRAJANewReduce(VariantID vid);

  template < size_t block_size, typename MappingHelper >
  void runHipVariantBase(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runHipVariantRAJA(VariantID vid);
  template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
  void runHipVariantRAJANewReduce(VariantID vid);

  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_type m_dx;
  Real_type m_pi;
  Real_type m_pi_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
