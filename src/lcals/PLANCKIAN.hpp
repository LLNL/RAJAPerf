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
/// PLANCKIAN kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] = u[i] / v[i];
///   w[i] = x[i] / ( exp( y[i] ) - 1.0 );
/// }
///

#ifndef RAJAPerf_Basic_PLANCKIAN_HPP
#define RAJAPerf_Basic_PLANCKIAN_HPP


#define PLANCKIAN_BODY  \
  y[i] = u[i] / v[i]; \
  w[i] = x[i] / ( exp( y[i] ) - 1.0 );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class PLANCKIAN : public KernelBase
{
public:

  PLANCKIAN(const RunParams& params);

  ~PLANCKIAN();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_u;
  Real_ptr m_v;
  Real_ptr m_w;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
