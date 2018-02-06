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
/// TRAP_INT kernel reference implementation:
///
/// Real_type trap_int_func(Real_type x,
///                         Real_type y,
///                         Real_type xp,
///                         Real_type yp)
/// {
///    Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
///    denom = 1.0/sqrt(denom);
///    return denom;
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///    Real_type x = x0 + i*h;
///    sumx += trap_int_func(x, y, xp, yp);
/// }
///

#ifndef RAJAPerf_Basic_TRAP_INT_HPP
#define RAJAPerf_Basic_TRAP_INT_HPP


#define TRAP_INT_BODY \
  Real_type x = x0 + i*h; \
  sumx += trap_int_func(x, y, xp, yp);


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class TRAP_INT : public KernelBase
{
public:

  TRAP_INT(const RunParams& params);

  ~TRAP_INT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_type m_x0;
  Real_type m_xp;
  Real_type m_y;
  Real_type m_yp;
  Real_type m_h;
  Real_type m_sumx_init;

  Real_type m_sumx;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
