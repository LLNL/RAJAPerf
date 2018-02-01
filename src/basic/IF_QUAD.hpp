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
/// IF_QUAD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type s = b[i]*b[i] - 4.0*a[i]*c[i];
///   if ( s >= 0 ) {
///     s = sqrt(s);
///     x2[i] = (-b[i]+s)/(2.0*a[i]);
///     x1[i] = (-b[i]-s)/(2.0*a[i]);
///   } else {
///     x2[i] = 0.0;
///     x1[i] = 0.0;
///   }
/// }
///

#ifndef RAJAPerf_Basic_IF_QUAD_HPP
#define RAJAPerf_Basic_IF_QUAD_HPP

#include "common/KernelBase.hpp"


#define IF_QUAD_BODY  \
  Real_type s = b[i]*b[i] - 4.0*a[i]*c[i]; \
  if ( s >= 0 ) { \
    s = sqrt(s); \
    x2[i] = (-b[i]+s)/(2.0*a[i]); \
    x1[i] = (-b[i]-s)/(2.0*a[i]); \
  } else { \
    x2[i] = 0.0; \
    x1[i] = 0.0; \
  }


namespace rajaperf 
{
class RunParams;

namespace basic
{

class IF_QUAD : public KernelBase
{
public:

  IF_QUAD(const RunParams& params);

  ~IF_QUAD();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_ptr m_x1;
  Real_ptr m_x2;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
