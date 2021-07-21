//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
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

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runStdParVariant(VariantID vid);

private:
  Real_type m_dx;
  Real_type m_pi;
  Real_type m_pi_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
