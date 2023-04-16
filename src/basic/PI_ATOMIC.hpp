//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// PI_ATOMIC kernel reference implementation:
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

#ifndef RAJAPerf_Basic_PI_ATOMIC_HPP
#define RAJAPerf_Basic_PI_ATOMIC_HPP

#define PI_ATOMIC_DATA_SETUP \
  Real_type dx = m_dx; \
  Real_ptr pi = m_pi;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class PI_ATOMIC : public KernelBase
{
public:

  PI_ATOMIC(const RunParams& params);

  ~PI_ATOMIC();

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

  Real_type m_dx;
  Real_ptr m_pi;
  Real_type m_pi_init;
  Real_type m_pi_final;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
