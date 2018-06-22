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
/// EOS kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   x[i] = u[i] + r*( z[i] + r*y[i] ) +
///                 t*( u[i+3] + r*( u[i+2] + r*u[i+1] ) +
///                    t*( u[i+6] + q*( u[i+5] + q*u[i+4] ) ) );
/// }
///

#ifndef RAJAPerf_Basic_EOS_HPP
#define RAJAPerf_Basic_EOS_HPP


#define EOS_BODY  \
  x[i] = u[i] + r*( z[i] + r*y[i] ) + \
                t*( u[i+3] + r*( u[i+2] + r*u[i+1] ) + \
                   t*( u[i+6] + q*( u[i+5] + q*u[i+4] ) ) );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class EOS : public KernelBase
{
public:

  EOS(const RunParams& params);

  ~EOS();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_ptr m_u;

  Real_type m_q;
  Real_type m_r;
  Real_type m_t;

  Index_type m_array_length;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
