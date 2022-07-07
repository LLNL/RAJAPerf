//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

#define DEL_DOT_VEC_2D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr xdot = m_xdot; \
  Real_ptr ydot = m_ydot; \
  Real_ptr div = m_div; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  Real_ptr x1,x2,x3,x4 ; \
  Real_ptr y1,y2,y3,y4 ; \
  Real_ptr fx1,fx2,fx3,fx4 ; \
  Real_ptr fy1,fy2,fy3,fy4 ; \
\
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
class ADomain;

class DEL_DOT_VEC_2D : public KernelBase
{
public:

  DEL_DOT_VEC_2D(const RunParams& params);

  ~DEL_DOT_VEC_2D();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

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
