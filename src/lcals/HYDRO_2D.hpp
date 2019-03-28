//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HYDRO_2D kernel reference implementation:
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     za[k][j] = ( zp[k+1][j-1] +zq[k+1][j-1] -zp[k][j-1] -zq[k][j-1] )*
///                ( zr[k][j] +zr[k][j-1] ) / ( zm[k][j-1] +zm[k+1][j-1]);
///     zb[k][j] = ( zp[k][j-1] +zq[k][j-1] -zp[k][j] -zq[k][j] ) *
///                ( zr[k][j] +zr[k-1][j] ) / ( zm[k][j] +zm[k][j-1]);
///   }
/// }
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     zu[k][j] += s*( za[k][j]   *( zz[k][j] - zz[k][j+1] ) -
///                     za[k][j-1] *( zz[k][j] - zz[k][j-1] ) -
///                     zb[k][j]   *( zz[k][j] - zz[k-1][j] ) +
///                     zb[k+1][j] *( zz[k][j] - zz[k+1][j] ) );
///     zv[k][j] += s*( za[k][j]   *( zr[k][j] - zr[k][j+1] ) -
///                     za[k][j-1] *( zr[k][j] - zr[k][j-1] ) -
///                     zb[k][j]   *( zr[k][j] - zr[k-1][j] ) +
///                     zb[k+1][j] *( zr[k][j] - zr[k+1][j] ) );
///   }
/// }
///
/// for (Index_type k=1 ; k<kn-1 ; k++) {
///   for (Index_type j=1 ; j<jn-1 ; j++) {
///     zrout[k][j] = zr[k][j] + t*zu[k][j];
///     zzout[k][j] = zz[k][j] + t*zv[k][j];
///   }
/// }
///

#ifndef RAJAPerf_Basic_HYDRO_2D_HPP
#define RAJAPerf_Basic_HYDRO_2D_HPP


#define HYDRO_2D_BODY1  \
  za[j+k*jn] = ( zp[j-1+(k+1)*jn] + zq[j-1+(k+1)*jn] - zp[j-1+k*jn] - zq[j-1+k*jn] ) * \
               ( zr[j+k*jn] + zr[j-1+k*jn] ) / ( zm[j-1+k*jn] + zm[j-1+(k+1)*jn] ); \
  zb[j+k*jn] = ( zp[j-1+k*jn] + zq[j-1+k*jn] - zp[j+k*jn] - zq[j+k*jn] ) * \
               ( zr[j+k*jn] + zr[j+(k-1)*jn] ) / ( zm[j+k*jn] + zm[j-1+k*jn] );

#define HYDRO_2D_BODY2 \
  zu[j+k*jn] += s*( za[j+k*jn] * ( zz[j+k*jn] - zz[j+1+k*jn] ) - \
                    za[j-1+k*jn] * ( zz[j+k*jn] - zz[j-1+k*jn] ) - \
                    zb[j+k*jn] * ( zz[j+k*jn] - zz[j+(k-1)*jn] ) + \
                    zb[j+(k+1)*jn] * ( zz[j+k*jn] - zz[j+(k+1)*jn] ) ); \
  zv[j+k*jn] += s*( za[j+k*jn] * ( zr[j+k*jn] - zr[j+1+k*jn] ) - \
                    za[j-1+k*jn] * ( zr[j+k*jn] - zr[j-1+k*jn] ) - \
                    zb[j+k*jn] * ( zr[j+k*jn] - zr[j+(k-1)*jn] ) + \
                    zb[j+(k+1)*jn] * ( zr[j+k*jn] - zr[j+(k+1)*jn] ) );

#define HYDRO_2D_BODY3 \
  zrout[j+k*jn] = zr[j+k*jn] + t*zu[j+k*jn]; \
  zzout[j+k*jn] = zz[j+k*jn] + t*zv[j+k*jn]; \


#define HYDRO_2D_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, RAJA::Layout<2, Index_type, 1> >; \
\
  std::array<RAJA::idx_t, 2> view_perm {{0, 1}}; \
\
  VIEW_TYPE za(zadat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zb(zbdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zm(zmdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zp(zpdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zq(zqdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zr(zrdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zu(zudat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zv(zvdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zz(zzdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm)); \
  VIEW_TYPE zrout(zroutdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm));\
  VIEW_TYPE zzout(zzoutdat, RAJA::make_permuted_layout({{kn, jn}}, view_perm));

#define HYDRO_2D_BODY1_RAJA  \
  za(k,j) = ( zp(k+1,j-1) + zq(k+1,j-1) - zp(k,j-1) - zq(k,j-1) ) * \
            ( zr(k,j) + zr(k,j-1) ) / ( zm(k,j-1) + zm(k+1,j-1) ); \
  zb(k,j) = ( zp(k,j-1) + zq(k,j-1) - zp(k,j) - zq(k,j) ) * \
            ( zr(k,j) + zr(k-1,j) ) / ( zm(k,j) + zm(k,j-1));

#define HYDRO_2D_BODY2_RAJA \
  zu(k,j) += s*( za(k,j) * ( zz(k,j) - zz(k,j+1) ) - \
                 za(k,j-1) * ( zz(k,j) - zz(k,j-1) ) - \
                 zb(k,j) * ( zz(k,j) - zz(k-1,j) ) + \
                 zb(k+1,j) * ( zz(k,j) - zz(k+1,j) ) ); \
  zv(k,j) += s*( za(k,j) * ( zr(k,j) - zr(k,j+1) ) - \
                 za(k,j-1) * ( zr(k,j) - zr(k,j-1) ) - \
                 zb(k,j) * ( zr(k,j) - zr(k-1,j) ) + \
                 zb(k+1,j) * ( zr(k,j) - zr(k+1,j) ) );

#define HYDRO_2D_BODY3_RAJA \
  zrout(k,j) = zr(k,j) + t*zu(k,j); \
  zzout(k,j) = zz(k,j) + t*zv(k,j);



#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class HYDRO_2D : public KernelBase
{
public:

  HYDRO_2D(const RunParams& params);

  ~HYDRO_2D();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_za;
  Real_ptr m_zb;
  Real_ptr m_zm;
  Real_ptr m_zp;
  Real_ptr m_zq;
  Real_ptr m_zr;
  Real_ptr m_zu;
  Real_ptr m_zv;
  Real_ptr m_zz;

  Real_ptr m_zrout;
  Real_ptr m_zzout;

  Real_type m_s;
  Real_type m_t;

  Index_type m_jn;
  Index_type m_kn;

  Index_type m_array_length;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
