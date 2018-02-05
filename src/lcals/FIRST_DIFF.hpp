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
/// FIRST_DIFF kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   x[i] = y[i+1] - y[i];
/// }
///

#ifndef RAJAPerf_Basic_FIRST_DIFF_HPP
#define RAJAPerf_Basic_FIRST_DIFF_HPP


#define FIRST_DIFF_BODY  \
  x[i] = y[i+1] - y[i];


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class FIRST_DIFF : public KernelBase
{
public:

  FIRST_DIFF(const RunParams& params);

  ~FIRST_DIFF();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;

  Index_type m_array_length;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
