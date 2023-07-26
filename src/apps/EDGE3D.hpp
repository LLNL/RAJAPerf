//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// EDGE3D kernel reference implementation:
///
/// NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
/// NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;
///
/// for (Index_type i = ibegin ; i < iend ; ++i ) {
///
///   double x[NB] = {x0[i],x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i]};
///   double y[NB] = {y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i]};
///   double z[NB] = {z0[i],z1[i],z2[i],z3[i],z4[i],z5[i],z6[i],z7[i]};
///   double edge_matrix[EB][EB];
///
  // Get integration points and weights
///   double qpts_1d[MAX_QUAD_ORDER];
///   double wgts_1d[MAX_QUAD_ORDER];
///
///   get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);
///
  // Compute cell centered Jacobian
///   const double jxx_cc = Jxx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jxy_cc = Jxy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jxz_cc = Jxz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyx_cc = Jyx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyy_cc = Jyy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jyz_cc = Jyz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzx_cc = Jzx(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzy_cc = Jzy(x, y, z, 0.25, 0.25, 0.25, 0.25);
///   const double jzz_cc = Jzz(x, y, z, 0.25, 0.25, 0.25, 0.25);
///
  // Compute cell centered Jacobian determinant
///   const double detj_cc = compute_detj(
///     jxx_cc, jxy_cc, jxz_cc,
///     jyx_cc, jyy_cc, jyz_cc,
///     jzx_cc, jzy_cc, jzz_cc);
///
  // Initialize the stiffness matrix
///   for (int m = 0; m < EB; m++) {
///     for (int p = m; p < EB; p++) {
///       matrix[m][p] = 0.0;
///     }
///   }
///
  // Compute values at each quadrature point
///   for ( int i = 0; i < quad_order; i++ ) {
///
///     const double xloc = qpts_1d[i];
///     const double tmpx = 1. - xloc;
///
///     double dbasisx[EB] = {0};
///     curl_edgebasis_x(dbasisx, tmpx, xloc);
///
///     for ( int j = 0; j < quad_order; j++ ) {
///
///       const double yloc = qpts_1d[j];
///       const double wgtxy = wgts_1d[i]*wgts_1d[j];
///       const double tmpy = 1. - yloc;
///
///       double tmpxy    = tmpx*tmpy;
///       double xyloc    = xloc*yloc;
///       double tmpxyloc = tmpx*yloc;
///       double xloctmpy = xloc*tmpy;
///
///       const double jzx = Jzx(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///       const double jzy = Jzy(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///       const double jzz = Jzz(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
///
///       double ebasisz[EB] = {0};
///       edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);
///
///       double dbasisy[EB] = {0};
///       curl_edgebasis_y(dbasisy, tmpy, yloc);
///
      // Differeniate basis with respect to z at this quadrature point
///
///       for ( int k = 0; k < quad_order; k++ ) {
///
///         const double zloc = qpts_1d[k];
///         const double wgts = wgtxy*wgts_1d[k];
///         const double tmpz = 1. - zloc;
///
///         const double tmpxz    = tmpx*tmpz;
///         const double tmpyz    = tmpy*tmpz;
///
///         const double xzloc    = xloc*zloc;
///         const double yzloc    = yloc*zloc;
///
///         const double tmpyzloc = tmpy*zloc;
///         const double tmpxzloc = tmpx*zloc;
///
///         const double yloctmpz = yloc*tmpz;
///         const double xloctmpz = xloc*tmpz;
///
///         const double jxx = Jxx(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jxy = Jxy(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jxz = Jxz(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
///         const double jyx = Jyx(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///         const double jyy = Jyy(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///         const double jyz = Jyz(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
///
///         double jinvxx, jinvxy, jinvxz,
///                jinvyx, jinvyy, jinvyz,
///                jinvzx, jinvzy, jinvzz,
///                detj_unfixed, detj, abs_detj, invdetj;
///
///         jacobian_inv(
///           jxx, jxy, jxz,
///           jyx, jyy, jyz,
///           jzx, jzy, jzz,
///           detj_cc, detj_tol,
///           jinvxx, jinvxy, jinvxz,
///           jinvyx, jinvyy, jinvyz,
///           jinvzx, jinvzy, jinvzz,
///           detj_unfixed, detj, abs_detj, invdetj);
///
///         const double detjwgts = wgts*abs_detj;
///
///         double ebasisx[EB] = {0};
///         edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);
///
///         double ebasisy[EB] = {0};
///         edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);
///
///         double dbasisz[EB] = {0};
///         curl_edgebasis_z(dbasisz, tmpz, zloc);
///
///         const double inv_abs_detj = 1./(abs_detj+ptiny);
///
///         double tebasisx[EB] = {0};
///         double tebasisy[EB] = {0};
///         double tebasisz[EB] = {0};
///
///         transform_edge_basis(
///           jinvxx, jinvxy, jinvxz,
///           jinvyx, jinvyy, jinvyz,
///           jinvzx, jinvzy, jinvzz,
///           ebasisx, ebasisy, ebasisz,
///           tebasisx, tebasisy, tebasisz);
///
///         double tdbasisx[EB] = {0};
///         double tdbasisy[EB] = {0};
///         double tdbasisz[EB] = {0};
///
///         transform_curl_edge_basis(
///           jxx, jxy, jxz,
///           jyx, jyy, jyz,
///           jzx, jzy, jzz,
///           inv_abs_detj,
///           dbasisx, dbasisy, dbasisz,
///           tdbasisx, tdbasisy, tdbasisz);
///
        // the inner product: alpha*<w_i, w_j>
///         inner_product(
///           detjwgts*alpha,
///           tebasisx, tebasisy, tebasisz,
///           tebasisx, tebasisy, tebasisz,
///           matrix, true);
///
         // the inner product: beta*<Curl(w_i), Curl(w_j)>
///         inner_product(
///           detjwgts*beta,
///           tdbasisx, tdbasisy, tdbasisz,
///           tdbasisx, tdbasisy, tdbasisz,
///           matrix, true);
///
///       }
///     }
///   }
///   sum[i] = 0.0;
///   for (int m = 0; m < EB; m++) {
///     Real_type check = 0.0;
///     for (int p = 0; p < EB; p++) {
///       check += edge_matrix[m*EB + p];
///     }
///     sum[i] += check;
///   }
/// }

#ifndef RAJAPerf_Apps_EDGE3D_HPP
#define RAJAPerf_Apps_EDGE3D_HPP

#define NQ_1D 2

#include "mixed_fem_helper.hpp"

RAJA_HOST_DEVICE
RAJA_INLINE void edge_MpSmatrix(
  const rajaperf::Real_type  (&x)[NB],
  const rajaperf::Real_type  (&y)[NB],
  const rajaperf::Real_type  (&z)[NB],
  rajaperf::Real_type        alpha,
  rajaperf::Real_type        beta,
  const rajaperf::Real_type  detj_tol,
  const rajaperf::Int_type   quad_type,
  const rajaperf::Int_type   quad_order,
  rajaperf::Real_type        (&matrix)[EB][EB])
{
  // Get integration points and weights
  rajaperf::Real_type qpts_1d[MAX_QUAD_ORDER];
  rajaperf::Real_type wgts_1d[MAX_QUAD_ORDER];

  get_quadrature_rule(quad_type, quad_order, qpts_1d, wgts_1d);

  // Compute cell centered Jacobian
  const rajaperf::Real_type jxx_cc = Jxx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jxy_cc = Jxy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jxz_cc = Jxz(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyx_cc = Jyx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyy_cc = Jyy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jyz_cc = Jyz(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzx_cc = Jzx(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzy_cc = Jzy(x, y, z, 0.25, 0.25, 0.25, 0.25);
  const rajaperf::Real_type jzz_cc = Jzz(x, y, z, 0.25, 0.25, 0.25, 0.25);

  // Compute cell centered Jacobian determinant
  const rajaperf::Real_type detj_cc = compute_detj(
    jxx_cc, jxy_cc, jxz_cc,
    jyx_cc, jyy_cc, jyz_cc,
    jzx_cc, jzy_cc, jzz_cc);

  // Initialize the stiffness matrix
  for (rajaperf::Int_type m = 0; m < EB; m++) {
    for (rajaperf::Int_type p = m; p < EB; p++) {
      matrix[m][p] = 0.0;
    }
  }

  // Compute values at each quadrature point
  for ( rajaperf::Int_type i = 0; i < quad_order; i++ ) {

    const rajaperf::Real_type xloc = qpts_1d[i];
    const rajaperf::Real_type tmpx = 1. - xloc;

    rajaperf::Real_type dbasisx[EB] = {0};
    curl_edgebasis_x(dbasisx, tmpx, xloc);

    for ( rajaperf::Int_type j = 0; j < quad_order; j++ ) {

      const rajaperf::Real_type yloc = qpts_1d[j];
      const rajaperf::Real_type wgtxy = wgts_1d[i]*wgts_1d[j];
      const rajaperf::Real_type tmpy = 1. - yloc;

      rajaperf::Real_type tmpxy    = tmpx*tmpy;
      rajaperf::Real_type xyloc    = xloc*yloc;
      rajaperf::Real_type tmpxyloc = tmpx*yloc;
      rajaperf::Real_type xloctmpy = xloc*tmpy;

      const rajaperf::Real_type jzx = Jzx(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
      const rajaperf::Real_type jzy = Jzy(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);
      const rajaperf::Real_type jzz = Jzz(x, y, z, tmpxy, xloctmpy, xyloc, tmpxyloc);

      rajaperf::Real_type ebasisz[EB] = {0};
      edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);

      rajaperf::Real_type dbasisy[EB] = {0};
      curl_edgebasis_y(dbasisy, tmpy, yloc);

      // Differeniate basis with respect to z at this quadrature point

      for ( rajaperf::Int_type k = 0; k < quad_order; k++ ) {

        const rajaperf::Real_type zloc = qpts_1d[k];
        const rajaperf::Real_type wgts = wgtxy*wgts_1d[k];
        const rajaperf::Real_type tmpz = 1. - zloc;

        const rajaperf::Real_type tmpxz    = tmpx*tmpz;
        const rajaperf::Real_type tmpyz    = tmpy*tmpz;

        const rajaperf::Real_type xzloc    = xloc*zloc;
        const rajaperf::Real_type yzloc    = yloc*zloc;

        const rajaperf::Real_type tmpyzloc = tmpy*zloc;
        const rajaperf::Real_type tmpxzloc = tmpx*zloc;

        const rajaperf::Real_type yloctmpz = yloc*tmpz;
        const rajaperf::Real_type xloctmpz = xloc*tmpz;

        const rajaperf::Real_type jxx = Jxx(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jxy = Jxy(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jxz = Jxz(x, y, z, tmpyz, yloctmpz, tmpyzloc, yzloc);
        const rajaperf::Real_type jyx = Jyx(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
        const rajaperf::Real_type jyy = Jyy(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);
        const rajaperf::Real_type jyz = Jyz(x, y, z, tmpxz, xloctmpz, tmpxzloc, xzloc);

        rajaperf::Real_type jinvxx, jinvxy, jinvxz,
                            jinvyx, jinvyy, jinvyz,
                            jinvzx, jinvzy, jinvzz,
                            detj_unfixed, detj, abs_detj, invdetj;

        jacobian_inv(
          jxx, jxy, jxz,
          jyx, jyy, jyz,
          jzx, jzy, jzz,
          detj_cc, detj_tol,
          jinvxx, jinvxy, jinvxz,
          jinvyx, jinvyy, jinvyz,
          jinvzx, jinvzy, jinvzz,
          detj_unfixed, detj, abs_detj, invdetj);

        const rajaperf::Real_type detjwgts = wgts*abs_detj;

        rajaperf::Real_type ebasisx[EB] = {0};
        edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);

        rajaperf::Real_type ebasisy[EB] = {0};
        edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);

        rajaperf::Real_type dbasisz[EB] = {0};
        curl_edgebasis_z(dbasisz, tmpz, zloc);

        const rajaperf::Real_type inv_abs_detj = 1./(abs_detj+ptiny);

        rajaperf::Real_type tebasisx[EB] = {0};
        rajaperf::Real_type tebasisy[EB] = {0};
        rajaperf::Real_type tebasisz[EB] = {0};

        transform_edge_basis(
          jinvxx, jinvxy, jinvxz,
          jinvyx, jinvyy, jinvyz,
          jinvzx, jinvzy, jinvzz,
          ebasisx, ebasisy, ebasisz,
          tebasisx, tebasisy, tebasisz);

        rajaperf::Real_type tdbasisx[EB] = {0};
        rajaperf::Real_type tdbasisy[EB] = {0};
        rajaperf::Real_type tdbasisz[EB] = {0};

        transform_curl_edge_basis(
          jxx, jxy, jxz,
          jyx, jyy, jyz,
          jzx, jzy, jzz,
          inv_abs_detj,
          dbasisx, dbasisy, dbasisz,
          tdbasisx, tdbasisy, tdbasisz);

        // the inner product: alpha*<w_i, w_j>
        inner_product(
          detjwgts*alpha,
          tebasisx, tebasisy, tebasisz,
          tebasisx, tebasisy, tebasisz,
          matrix, true);

         // the inner product: beta*<Curl(w_i), Curl(w_j)>
        inner_product(
          detjwgts*beta,
          tdbasisx, tdbasisy, tdbasisz,
          tdbasisx, tdbasisy, tdbasisz,
          matrix, true);
      }
    }
  }
}

#define EDGE3D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
  Real_ptr sum = m_sum; \
\
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  Real_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  Real_ptr z0,z1,z2,z3,z4,z5,z6,z7 ; \
\
  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ; \
  NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ; \
  NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

#define EDGE3D_BODY \
  rajaperf::Real_type X[NB] = {x0[i],x1[i],x2[i],x3[i],x4[i],x5[i],x6[i],x7[i]};\
  rajaperf::Real_type Y[NB] = {y0[i],y1[i],y2[i],y3[i],y4[i],y5[i],y6[i],y7[i]};\
  rajaperf::Real_type Z[NB] = {z0[i],z1[i],z2[i],z3[i],z4[i],z5[i],z6[i],z7[i]};\
  rajaperf::Real_type edge_matrix[EB][EB];\
  edge_MpSmatrix(X, Y, Z, 1.0, 1.0, 0.0, 1.0, NQ_1D, edge_matrix);\
  rajaperf::Real_type local_sum = 0.0;\
  for (rajaperf::Int_type m = 0; m < EB; m++) {\
    rajaperf::Real_type check = 0.0;\
    for (rajaperf::Int_type p = 0; p < EB; p++) {\
      check += edge_matrix[m][p];\
    }\
    local_sum += check;\
  }\
  sum[i] = local_sum;\

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{
class ADomain;

class EDGE3D : public KernelBase
{
public:

  EDGE3D(const RunParams& params);

  ~EDGE3D();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

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
  Real_ptr m_z;
  Real_ptr m_sum;

  ADomain* m_domain;
  Index_type m_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
