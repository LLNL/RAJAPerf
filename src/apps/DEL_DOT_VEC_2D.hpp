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
/// DEL_DOT_VEC_2D kernel reference implementation:
///
/// for (Index_type ii = ibegin; ii < iend; ++ii ) {
///   Index_type i = real_zones[ii];
///
///   Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ;
///   Real_type xj  = half * ( x2[i]  + x3[i]  - x4[i]  - x1[i]  ) ;
///
///   Real_type yi  = half * ( y1[i]  + y2[i]  - y3[i]  - y4[i]  ) ;
///   Real_type yj  = half * ( y2[i]  + y3[i]  - y4[i]  - y1[i]  ) ;
///
///   Real_type fxi = half * ( fx1[i] + fx2[i] - fx3[i] - fx4[i] ) ;
///   Real_type fxj = half * ( fx2[i] + fx3[i] - fx4[i] - fx1[i] ) ;
///
///   Real_type fyi = half * ( fy1[i] + fy2[i] - fy3[i] - fy4[i] ) ;
///   Real_type fyj = half * ( fy2[i] + fy3[i] - fy4[i] - fy1[i] ) ;
///
///   Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ;
///
///   Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ;
///
///   Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ;
///
///   Real_type affine = ( fy1[i] + fy2[i] + fy3[i] + fy4[i] ) /
///                      ( y1[i]  + y2[i]  + y3[i]  + y4[i]  ) ;
///
///   div[i] = dfxdx + dfydy + affine ;
/// }
///

#ifndef RAJAPerf_Apps_DEL_DOT_VEC_2D_HPP
#define RAJAPerf_Apps_DEL_DOT_VEC_2D_HPP


#define DEL_DOT_VEC_2D_DATA_INDEX \
  Index_ptr real_zones = m_domain->real_zones;

#define DEL_DOT_VEC_2D_BODY_INDEX \
  Index_type i = real_zones[ii];

#define DEL_DOT_VEC_2D_BODY \
\
  Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ; \
  Real_type xj  = half * ( x2[i]  + x3[i]  - x4[i]  - x1[i]  ) ; \
 \
  Real_type yi  = half * ( y1[i]  + y2[i]  - y3[i]  - y4[i]  ) ; \
  Real_type yj  = half * ( y2[i]  + y3[i]  - y4[i]  - y1[i]  ) ; \
 \
  Real_type fxi = half * ( fx1[i] + fx2[i] - fx3[i] - fx4[i] ) ; \
  Real_type fxj = half * ( fx2[i] + fx3[i] - fx4[i] - fx1[i] ) ; \
 \
  Real_type fyi = half * ( fy1[i] + fy2[i] - fy3[i] - fy4[i] ) ; \
  Real_type fyj = half * ( fy2[i] + fy3[i] - fy4[i] - fy1[i] ) ; \
 \
  Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ; \
 \
  Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ; \
 \
  Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ; \
 \
  Real_type affine = ( fy1[i] + fy2[i] + fy3[i] + fy4[i] ) / \
                     ( y1[i]  + y2[i]  + y3[i]  + y4[i]  ) ; \
 \
  div[i] = dfxdx + dfydy + affine ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{
struct ADomain;

class DEL_DOT_VEC_2D : public KernelBase
{
public:

  DEL_DOT_VEC_2D(const RunParams& params);

  ~DEL_DOT_VEC_2D();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_xdot;
  Real_ptr m_ydot;
  Real_ptr m_div;

  Real_type m_ptiny;
  Real_type m_half;

  ADomain* m_domain;
  Index_type m_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
