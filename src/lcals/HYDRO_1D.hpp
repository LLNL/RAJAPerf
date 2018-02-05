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
/// HYDRO_1D kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );
/// }
///

#ifndef RAJAPerf_Basic_HYDRO_1D_HPP
#define RAJAPerf_Basic_HYDRO_1D_HPP


#define HYDRO_1D_BODY  \
  x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class HYDRO_1D : public KernelBase
{
public:

  HYDRO_1D(const RunParams& params);

  ~HYDRO_1D();

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

  Real_type m_q;
  Real_type m_r;
  Real_type m_t;

  Index_type m_array_length; 
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
