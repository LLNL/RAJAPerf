//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT_VIEW1D kernel reference implementation:
///
/// const Real_type val = ...;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = val;
/// }
///
/// RAJA variants use a "View" and "Layout" to do the same thing. These 
/// RAJA constructs provide little benfit in 1D, but they are used here
/// to exercise those RAJA mechanics in the simplest scenario.
///

#ifndef RAJAPerf_Basic_INIT_VIEW1D_HPP
#define RAJAPerf_Basic_INIT_VIEW1D_HPP


#define INIT_VIEW1D_BODY  \
  a[i] = v;

#define INIT_VIEW1D_BODY_RAJA  \
  view(i) = v;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class INIT_VIEW1D : public KernelBase
{
public:

  INIT_VIEW1D(const RunParams& params);

  ~INIT_VIEW1D();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_type m_val;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
