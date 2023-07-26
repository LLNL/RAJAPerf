//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef MIXED_FEM_HELPER
#define MIXED_FEM_HELPER

#include "RAJA/RAJA.hpp"

#include "common/RPTypes.hpp"

#include <cmath>

#define NB 8
#define EB 12
#define FB 6
#define MAX_QUAD_ORDER 5

constexpr rajaperf::Real_type ptiny = 1.0e-50;

//
// Common FEM functions
//

RAJA_HOST_DEVICE
RAJA_INLINE void LinAlg_qrule_Lobatto(
  rajaperf::Int_type    order,
  rajaperf::Real_type *qpts1D,
  rajaperf::Real_type *wgts1D)
{
  // Define the Gauss-Lobatto quadrature points and weights over the
  // 1D domain [0,1] for rules up to order 5
  switch( order ) {

  case 1 :

    // Order 1 Gauss-Lobatto Points
    qpts1D[0] = 0.5;

    // Order 1 Gauss-Lobatto Weights
    wgts1D[0] = 1.0;

    break;

  case 2 :

    // Order 2 Gauss-Lobatto Points
    qpts1D[0] = 0.0;
    qpts1D[1] = 1.0;

    // Order 2 Gauss-Lobatto Weights
    wgts1D[0] = 0.5;
    wgts1D[1] = 0.5;

    break;

  case 3 :

    // Order 3 Gauss-Lobatto Points
    qpts1D[0] = 0.0;
    qpts1D[1] = 0.5;
    qpts1D[2] = 1.0;

    // Order 3 Gauss-Lobatto Weights
    wgts1D[0] = 0.166666666666666667;
    wgts1D[1] = 0.666666666666666667;
    wgts1D[2] = 0.166666666666666667;

    break;

  case 4 :

    // Order 4 Gauss-Lobatto Points
    qpts1D[0] = 0.0;
    qpts1D[1] = 0.276393202250021030;
    qpts1D[2] = 0.723606797749978970;
    qpts1D[3] = 1.0;

    // Order 4 Gauss-Lobatto Weights
    wgts1D[0] = 0.0833333333333333333;
    wgts1D[1] = 0.416666666666666667;
    wgts1D[2] = 0.416666666666666667;
    wgts1D[3] = 0.0833333333333333333;

    break;

  case 5 :

    // Order 5 Gauss-Lobatto Points
    qpts1D[0] = 0.0;
    qpts1D[1] = 0.172673164646011428;
    qpts1D[2] = 0.5;
    qpts1D[3] = 0.827326835353988572;
    qpts1D[4] = 1.0;

    // Order 5 Gauss-Lobatto Weights
    wgts1D[0] = 0.05;
    wgts1D[1] = 0.272222222222222222;
    wgts1D[2] = 0.355555555555555556;
    wgts1D[3] = 0.272222222222222222;
    wgts1D[4] = 0.05;

    break;

  }

}

RAJA_HOST_DEVICE
RAJA_INLINE void LinAlg_qrule_Legendre(
  rajaperf::Int_type    order,
  rajaperf::Real_type *qpts1D,
  rajaperf::Real_type *wgts1D)
{
  // Define the Gauss-Legendre quadrature points and weights over the
  // 1D domain [0,1] for rules up to order 5
  switch( order ) {

  case 1 :

    // Order 1 Gauss-Legendre Points
    qpts1D[0] = 0.5;

    // Order 1 Gauss-Legendre Weights
    wgts1D[0] = 1.0;

    break;

  case 2 :

    // Order 2 Gauss-Legendre Points
    qpts1D[0] = 0.211324865405187118;
    qpts1D[1] = 0.788675134594812882;

    // Order 2 Gauss-Legendre Weights
    wgts1D[0] = 0.5;
    wgts1D[1] = 0.5;

    break;

  case 3 :

    // Order 3 Gauss-Legendre Points
    qpts1D[0] = 0.112701665379258311;
    qpts1D[1] = 0.5;
    qpts1D[2] = 0.887298334620741689;

    // Order 3 Gauss-Legendre Weights
    wgts1D[0] = 0.277777777777777778;
    wgts1D[1] = 0.444444444444444444;
    wgts1D[2] = 0.277777777777777778;

    break;

  case 4 :

    // Order 4 Gauss-Legendre Points
    qpts1D[0] = 0.0694318442029737124;
    qpts1D[1] = 0.330009478207571868;
    qpts1D[2] = 0.669990521792428132;
    qpts1D[3] = 0.930568155797026288;

    // Order 4 Gauss-Legendre Weights
    wgts1D[0] = 0.173927422568726929;
    wgts1D[1] = 0.326072577431273071;
    wgts1D[2] = 0.326072577431273071;
    wgts1D[3] = 0.173927422568726929;

    break;

  case 5 :

    // Order 5 Gauss-Legendre Points
    qpts1D[0] = 0.0469100770306680036;
    qpts1D[1] = 0.230765344947158454;
    qpts1D[2] = 0.5;
    qpts1D[3] = 0.769234655052841546;
    qpts1D[4] = 0.953089922969331996;

    // Order 5 Gauss-Legendre Weights
    wgts1D[0] = 0.118463442528094544;
    wgts1D[1] = 0.239314335249683234;
    wgts1D[2] = 0.284444444444444444;
    wgts1D[3] = 0.239314335249683234;
    wgts1D[4] = 0.118463442528094544;

    break;

  }

}

RAJA_HOST_DEVICE
RAJA_INLINE void get_quadrature_rule(
  const rajaperf::Int_type    quad_type,
  const rajaperf::Int_type    quad_order,
  rajaperf::Real_type       (&qpts_1d)[MAX_QUAD_ORDER],
  rajaperf::Real_type       (&wgts_1d)[MAX_QUAD_ORDER])
{
  // Generate the 1D set of points and weights over the interval [0,1]
  switch( quad_type ) {

  case 0 :
    LinAlg_qrule_Lobatto(quad_order, qpts_1d, wgts_1d);
    break;

  case 1 :
    LinAlg_qrule_Legendre(quad_order, qpts_1d, wgts_1d);
    break;

  }
}

constexpr rajaperf::Int_type flops_compute_detj()
{
  return 17;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type compute_detj(
  const rajaperf::Real_type jxx,
  const rajaperf::Real_type jxy,
  const rajaperf::Real_type jxz,
  const rajaperf::Real_type jyx,
  const rajaperf::Real_type jyy,
  const rajaperf::Real_type jyz,
  const rajaperf::Real_type jzx,
  const rajaperf::Real_type jzy,
  const rajaperf::Real_type jzz)
{
  return
    jxy*jyz*jzx - jxz*jyy*jzx + jxz*jyx*jzy -
    jxx*jyz*jzy - jxy*jyx*jzz + jxx*jyy*jzz;
}

template<rajaperf::Int_type M>
RAJA_HOST_DEVICE
constexpr void transform_basis(
  const rajaperf::Real_type txx,
  const rajaperf::Real_type txy,
  const rajaperf::Real_type txz,
  const rajaperf::Real_type tyx,
  const rajaperf::Real_type tyy,
  const rajaperf::Real_type tyz,
  const rajaperf::Real_type tzx,
  const rajaperf::Real_type tzy,
  const rajaperf::Real_type tzz,
  const rajaperf::Real_type (&basis_x)[M],
  const rajaperf::Real_type (&basis_y)[M],
  const rajaperf::Real_type (&basis_z)[M],
  rajaperf::Real_type (&tbasis_x)[M],
  rajaperf::Real_type (&tbasis_y)[M],
  rajaperf::Real_type (&tbasis_z)[M])
{
  // Compute transformed basis function gradients
  for (rajaperf::Int_type m = 0; m < M; m++)
  {
    tbasis_x[m] = txx*basis_x[m] + txy*basis_y[m] + txz*basis_z[m];
    tbasis_y[m] = tyx*basis_x[m] + tyy*basis_y[m] + tyz*basis_z[m];
    tbasis_z[m] = tzx*basis_x[m] + tzy*basis_y[m] + tzz*basis_z[m];
  }
}

template<rajaperf::Int_type M, rajaperf::Int_type P>
constexpr rajaperf::Int_type flops_inner_product(const bool is_symmetric)
{
  return is_symmetric ? 6*P*(M+1)/2 : 6*P*M;
}

template<rajaperf::Int_type M, rajaperf::Int_type P>
RAJA_HOST_DEVICE
constexpr void inner_product(
  const rajaperf::Real_type weight,
  const rajaperf::Real_type (&basis_1_x)[M],
  const rajaperf::Real_type (&basis_1_y)[M],
  const rajaperf::Real_type (&basis_1_z)[M],
  const rajaperf::Real_type (&basis_2_x)[P],
  const rajaperf::Real_type (&basis_2_y)[P],
  const rajaperf::Real_type (&basis_2_z)[P],
  rajaperf::Real_type (&matrix)[P][M],
  const bool is_symmetric)
{
  // inner product is <basis_2, basis_1>
  for (rajaperf::Int_type p = 0; p < P; p++) {

    const rajaperf::Real_type txi = basis_2_x[p];
    const rajaperf::Real_type tyi = basis_2_y[p];
    const rajaperf::Real_type tzi = basis_2_z[p];

    const rajaperf::Int_type m0 = (is_symmetric && (M == P)) ? p : 0;

    for (rajaperf::Int_type m = m0; m < M; m++) {

      const rajaperf::Real_type txj = basis_1_x[m];
      const rajaperf::Real_type tyj = basis_1_y[m];
      const rajaperf::Real_type tzj = basis_1_z[m];

      matrix[p][m] += weight*(txi*txj + tyi*tyj + tzi*tzj);

      if(is_symmetric && (M == P) && (m > m0))
      {
        matrix[m][p] = matrix[p][m];
      }
    }
  }
}

constexpr rajaperf::Int_type flops_bad_zone_algorithm()
{
  return 5;
}

RAJA_HOST_DEVICE
RAJA_INLINE void bad_zone_algorithm(
  const rajaperf::Real_type detj_unfixed,
  const rajaperf::Real_type detj_cc,
  const rajaperf::Real_type detj_tol,
  rajaperf::Real_type& detj,
  rajaperf::Real_type& abs_detj,
  rajaperf::Real_type& inv_detj)
{
  detj = (fabs( detj_unfixed/detj_cc ) < detj_tol) ?
                               detj_cc : detj_unfixed ;
  abs_detj = fabs(detj);

  // Note that this uses a potentially negative detj

  inv_detj = 1.0/(detj + ptiny);
}

constexpr rajaperf::Int_type flops_jacobian_inv()
{
  return flops_compute_detj() + flops_bad_zone_algorithm() + 4*9;
}

RAJA_HOST_DEVICE
RAJA_INLINE void jacobian_inv(
  const rajaperf::Real_type jxx,
  const rajaperf::Real_type jxy,
  const rajaperf::Real_type jxz,
  const rajaperf::Real_type jyx,
  const rajaperf::Real_type jyy,
  const rajaperf::Real_type jyz,
  const rajaperf::Real_type jzx,
  const rajaperf::Real_type jzy,
  const rajaperf::Real_type jzz,
  const rajaperf::Real_type detj_cc,
  const rajaperf::Real_type detj_tol,
  rajaperf::Real_type &jinvxx,
  rajaperf::Real_type &jinvxy,
  rajaperf::Real_type &jinvxz,
  rajaperf::Real_type &jinvyx,
  rajaperf::Real_type &jinvyy,
  rajaperf::Real_type &jinvyz,
  rajaperf::Real_type &jinvzx,
  rajaperf::Real_type &jinvzy,
  rajaperf::Real_type &jinvzz,
  rajaperf::Real_type &detj_unfixed,
  rajaperf::Real_type &detj,
  rajaperf::Real_type &abs_detj,
  rajaperf::Real_type &inv_detj)
{
  // Compute determinant of Jacobian matrix at this quadrature point
  detj_unfixed = compute_detj(jxx, jxy, jxz,
                             jyx, jyy, jyz,
                             jzx, jzy, jzz);

  bad_zone_algorithm(detj_unfixed, detj_cc, detj_tol, detj, abs_detj, inv_detj);

  jinvxx =  (jyy*jzz - jyz*jzy)*inv_detj;
  jinvxy =  (jxz*jzy - jxy*jzz)*inv_detj;
  jinvxz =  (jxy*jyz - jxz*jyy)*inv_detj;

  jinvyx =  (jyz*jzx - jyx*jzz)*inv_detj;
  jinvyy =  (jxx*jzz - jxz*jzx)*inv_detj;
  jinvyz =  (jxz*jyx - jxx*jyz)*inv_detj;

  jinvzx =  (jyx*jzy - jyy*jzx)*inv_detj;
  jinvzy =  (jxy*jzx - jxx*jzy)*inv_detj;
  jinvzz =  (jxx*jyy - jxy*jyx)*inv_detj;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jzx(
  const rajaperf::Real_type  (&x)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpxy,
  const rajaperf::Real_type  xloctmpy,
  const rajaperf::Real_type  xyloc,
  const rajaperf::Real_type  tmpxyloc)
{
  return (x[4] - x[0])*tmpxy    +
         (x[5] - x[1])*xloctmpy +
         (x[6] - x[2])*xyloc    +
         (x[7] - x[3])*tmpxyloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jzy(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&y)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpxy,
  const rajaperf::Real_type  xloctmpy,
  const rajaperf::Real_type  xyloc,
  const rajaperf::Real_type  tmpxyloc)
{
  return (y[4] - y[0])*tmpxy    +
         (y[5] - y[1])*xloctmpy +
         (y[6] - y[2])*xyloc    +
         (y[7] - y[3])*tmpxyloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jzz(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&z)[NB],
  const rajaperf::Real_type  tmpxy,
  const rajaperf::Real_type  xloctmpy,
  const rajaperf::Real_type  xyloc,
  const rajaperf::Real_type  tmpxyloc)
{
  return (z[4] - z[0])*tmpxy    +
         (z[5] - z[1])*xloctmpy +
         (z[6] - z[2])*xyloc    +
         (z[7] - z[3])*tmpxyloc;
}

constexpr rajaperf::Int_type flops_Jxx()
{
  return 8;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jxx(
  const rajaperf::Real_type  (&x)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpyz,
  const rajaperf::Real_type  yloctmpz,
  const rajaperf::Real_type  tmpyzloc,
  const rajaperf::Real_type  yzloc)
{
  return (x[1] - x[0])*tmpyz   +
         (x[2] - x[3])*yloctmpz +
         (x[5] - x[4])*tmpyzloc +
         (x[6] - x[7])*yzloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jxy(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&y)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpyz,
  const rajaperf::Real_type  yloctmpz,
  const rajaperf::Real_type  tmpyzloc,
  const rajaperf::Real_type  yzloc)
{
  return (y[1] - y[0])*tmpyz    +
         (y[2] - y[3])*yloctmpz +
         (y[5] - y[4])*tmpyzloc +
         (y[6] - y[7])*yzloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jxz(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&z)[NB],
  const rajaperf::Real_type  tmpyz,
  const rajaperf::Real_type  yloctmpz,
  const rajaperf::Real_type  tmpyzloc,
  const rajaperf::Real_type  yzloc)
{
  return (z[1] - z[0])*tmpyz   +
         (z[2] - z[3])*yloctmpz +
         (z[5] - z[4])*tmpyzloc +
         (z[6] - z[7])*yzloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jyx(
  const rajaperf::Real_type  (&x)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpxz,
  const rajaperf::Real_type  xloctmpz,
  const rajaperf::Real_type  tmpxzloc,
  const rajaperf::Real_type  xzloc)
{
  return (x[3] - x[0])*tmpxz    +
         (x[2] - x[1])*xloctmpz +
         (x[7] - x[4])*tmpxzloc +
         (x[6] - x[5])*xzloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jyy(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&y)[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(z))[NB],
  const rajaperf::Real_type  tmpxz,
  const rajaperf::Real_type  xloctmpz,
  const rajaperf::Real_type  tmpxzloc,
  const rajaperf::Real_type  xzloc)
{
  return (y[3] - y[0])*tmpxz    +
         (y[2] - y[1])*xloctmpz +
         (y[7] - y[4])*tmpxzloc +
         (y[6] - y[5])*xzloc;
}

RAJA_HOST_DEVICE
constexpr rajaperf::Real_type Jyz(
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(x))[NB],
  const rajaperf::Real_type  (&RAJA_UNUSED_ARG(y))[NB],
  const rajaperf::Real_type  (&z)[NB],
  const rajaperf::Real_type  tmpxz,
  const rajaperf::Real_type  xloctmpz,
  const rajaperf::Real_type  tmpxzloc,
  const rajaperf::Real_type  xzloc)
{
  return (z[3] - z[0])*tmpxz    +
         (z[2] - z[1])*xloctmpz +
         (z[7] - z[4])*tmpxzloc +
         (z[6] - z[5])*xzloc;
}

//-----------------------------------------
// Node basis
//-----------------------------------------
RAJA_HOST_DEVICE
constexpr void nodebasis(
  rajaperf::Real_type (&basis)[NB],
  const rajaperf::Real_type tmpxy,
  const rajaperf::Real_type xloctmpy,
  const rajaperf::Real_type xyloc,
  const rajaperf::Real_type tmpxyloc,
  const rajaperf::Real_type zloc,
  const rajaperf::Real_type tmpz)
{
  basis[0] = tmpxy*tmpz;
  basis[1] = xloctmpy*tmpz;
  basis[2] = xyloc*tmpz;
  basis[3] = tmpxyloc*tmpz;
  basis[4] = tmpxy*zloc;
  basis[5] = xloctmpy*zloc;
  basis[6] = xyloc*zloc;
  basis[7] = tmpxyloc*zloc;
}

RAJA_HOST_DEVICE
constexpr void dnodebasis_dx(
  rajaperf::Real_type (&dbasis)[NB],
  const rajaperf::Real_type tmpyz,
  const rajaperf::Real_type yloctmpz,
  const rajaperf::Real_type tmpyzloc,
  const rajaperf::Real_type yzloc)
{
  dbasis[0] = -tmpyz;
  dbasis[1] =  tmpyz;
  dbasis[2] =  yloctmpz;
  dbasis[3] = -yloctmpz;
  dbasis[4] = -tmpyzloc;
  dbasis[5] =  tmpyzloc;
  dbasis[6] =  yzloc;
  dbasis[7] = -yzloc;
}

RAJA_HOST_DEVICE
constexpr void dnodebasis_dy(
  rajaperf::Real_type (&dbasis)[NB],
  const rajaperf::Real_type tmpxz,
  const rajaperf::Real_type xloctmpz,
  const rajaperf::Real_type tmpxzloc,
  const rajaperf::Real_type xzloc)
{
  dbasis[0] = -tmpxz;
  dbasis[1] = -xloctmpz;
  dbasis[2] =  xloctmpz;
  dbasis[3] =  tmpxz;
  dbasis[4] = -tmpxzloc;
  dbasis[5] = -xzloc;
  dbasis[6] =  xzloc;
  dbasis[7] =  tmpxzloc;
}

RAJA_HOST_DEVICE
constexpr void dnodebasis_dz(
  rajaperf::Real_type (&dbasis)[NB],
  const rajaperf::Real_type tmpxy,
  const rajaperf::Real_type xloctmpy,
  const rajaperf::Real_type xyloc,
  const rajaperf::Real_type tmpxyloc)
{
  dbasis[0] = -tmpxy;
  dbasis[1] = -xloctmpy;
  dbasis[2] = -xyloc;
  dbasis[3] = -tmpxyloc;
  dbasis[4] =  tmpxy;
  dbasis[5] =  xloctmpy;
  dbasis[6] =  xyloc;
  dbasis[7] =  tmpxyloc;
}

RAJA_HOST_DEVICE
constexpr void transform_node_dbasis(
  const rajaperf::Real_type jinvxx,
  const rajaperf::Real_type jinvxy,
  const rajaperf::Real_type jinvxz,
  const rajaperf::Real_type jinvyx,
  const rajaperf::Real_type jinvyy,
  const rajaperf::Real_type jinvyz,
  const rajaperf::Real_type jinvzx,
  const rajaperf::Real_type jinvzy,
  const rajaperf::Real_type jinvzz,
  rajaperf::Real_type (&basisx)[NB],
  rajaperf::Real_type (&basisy)[NB],
  rajaperf::Real_type (&basisz)[NB],
  rajaperf::Real_type (&tbasisx)[NB],
  rajaperf::Real_type (&tbasisy)[NB],
  rajaperf::Real_type (&tbasisz)[NB])
{
  // Transform is:  Grad(w_i) <- J^{-1} Grad(w_i)
  transform_basis(
    jinvxx, jinvxy, jinvxz,
    jinvyx, jinvyy, jinvyz,
    jinvzx, jinvzy, jinvzz,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}

//-----------------------------------------
// Edge basis
//-----------------------------------------
RAJA_HOST_DEVICE
constexpr void edgebasis_x(
  rajaperf::Real_type (&basisx)[EB],
  const rajaperf::Real_type tmpyz,
  const rajaperf::Real_type yloctmpz,
  const rajaperf::Real_type tmpyzloc,
  const rajaperf::Real_type yzloc)
{
  basisx[0]  = tmpyz;
  basisx[1]  = yloctmpz;
  basisx[2]  = tmpyzloc;
  basisx[3]  = yzloc;
  basisx[4]  = 0.0;
  basisx[5]  = 0.0;
  basisx[6]  = 0.0;
  basisx[7]  = 0.0;
  basisx[8]  = 0.0;
  basisx[9]  = 0.0;
  basisx[10] = 0.0;
  basisx[11] = 0.0;
}

// Evaluate basis with respect to y at this quadrature point
RAJA_HOST_DEVICE
constexpr void edgebasis_y(
  rajaperf::Real_type (&basisy)[EB],
  const rajaperf::Real_type tmpxz,
  const rajaperf::Real_type xloctmpz,
  const rajaperf::Real_type tmpxzloc,
  const rajaperf::Real_type xzloc)
{
  basisy[0]  = 0.0;
  basisy[1]  = 0.0;
  basisy[2]  = 0.0;
  basisy[3]  = 0.0;
  basisy[4]  = tmpxz;
  basisy[5]  = xloctmpz;
  basisy[6]  = tmpxzloc;
  basisy[7]  = xzloc;
  basisy[8]  = 0.0;
  basisy[9]  = 0.0;
  basisy[10] = 0.0;
  basisy[11] = 0.0;
}

// Evaluate basis with respect to z at this quadrature point
RAJA_HOST_DEVICE
constexpr void edgebasis_z(
  rajaperf::Real_type (&basisz)[EB],
  const rajaperf::Real_type tmpxy,
  const rajaperf::Real_type xloctmpy,
  const rajaperf::Real_type xyloc,
  const rajaperf::Real_type tmpxyloc)
{
  basisz[0]  = 0.0;
  basisz[1]  = 0.0;
  basisz[2]  = 0.0;
  basisz[3]  = 0.0;
  basisz[4]  = 0.0;
  basisz[5]  = 0.0;
  basisz[6]  = 0.0;
  basisz[7]  = 0.0;
  basisz[8]  = tmpxy;
  basisz[9]  = xloctmpy;
  basisz[10] = tmpxyloc;
  basisz[11] = xyloc;
}

// Differeniate basis with respect to x at this quadrature point
RAJA_HOST_DEVICE
constexpr void curl_edgebasis_x(
  rajaperf::Real_type (&dbasisx)[EB],
  const rajaperf::Real_type tmpx,
  const rajaperf::Real_type xpt)
{
  dbasisx[0]  =  0.0;  //
  dbasisx[1]  =  0.0;  //
  dbasisx[2]  =  0.0;  //
  dbasisx[3]  =  0.0;  //
  dbasisx[4]  =  tmpx; // +1*f0
  dbasisx[5]  =  xpt;  // +1*f1
  dbasisx[6]  = -tmpx; // -1*f0
  dbasisx[7]  = -xpt;  // -1*f1
  dbasisx[8]  = -tmpx; // -1*f0
  dbasisx[9]  = -xpt;  // -1*f1
  dbasisx[10] =  tmpx; // +1*f0
  dbasisx[11] =  xpt;  // +1*f1
}

// Differeniate basis with respect to y at this quadrature point
RAJA_HOST_DEVICE
constexpr void curl_edgebasis_y(
  rajaperf::Real_type (&dbasisy)[EB],
  const rajaperf::Real_type tmpy,
  const rajaperf::Real_type ypt)
{
  dbasisy[0]  = -tmpy; // -1*f2
  dbasisy[1]  = -ypt;  // -1*f3
  dbasisy[2]  =  tmpy; // +1*f2
  dbasisy[3]  =  ypt;  // +1*f3
  dbasisy[4]  =  0.0;  //
  dbasisy[5]  =  0.0;  //
  dbasisy[6]  =  0.0;  //
  dbasisy[7]  =  0.0;  //
  dbasisy[8]  =  tmpy; // +1*f2
  dbasisy[9]  = -tmpy; // -1*f2
  dbasisy[10] =  ypt;  // +1*f3
  dbasisy[11] = -ypt;  // -1*f3
}

// Differeniate basis with respect to z at this quadrature point
RAJA_HOST_DEVICE
constexpr void curl_edgebasis_z(
  rajaperf::Real_type (&dbasisz)[EB],
  const rajaperf::Real_type tmpz,
  const rajaperf::Real_type zpt)
{
  dbasisz[0]  =  tmpz; // +1*f4
  dbasisz[1]  = -tmpz; // -1*f4
  dbasisz[2]  =  zpt;  // +1*f5
  dbasisz[3]  = -zpt;  // -1*f5
  dbasisz[4]  = -tmpz; // -1*f4
  dbasisz[5]  =  tmpz; // +1*f4
  dbasisz[6]  = -zpt;  // -1*f5
  dbasisz[7]  =  zpt;  // +1 f5
  dbasisz[8]  =  0.0;  //
  dbasisz[9]  =  0.0;  //
  dbasisz[10] =  0.0;  //
  dbasisz[11] =  0.0;  //
}

RAJA_HOST_DEVICE
constexpr void edgebasis(
  const rajaperf::Real_type xloc,
  const rajaperf::Real_type yloc,
  const rajaperf::Real_type zloc,
  rajaperf::Real_type (&ebasisx)[EB],
  rajaperf::Real_type (&ebasisy)[EB],
  rajaperf::Real_type (&ebasisz)[EB])
{
  const rajaperf::Real_type tmpx = 1. - xloc;
  const rajaperf::Real_type tmpy = 1. - yloc;
  const rajaperf::Real_type tmpz = 1. - zloc;

  const rajaperf::Real_type tmpxy    = tmpx*tmpy;
  const rajaperf::Real_type xyloc    = xloc*yloc;
  const rajaperf::Real_type tmpxyloc = tmpx*yloc;
  const rajaperf::Real_type xloctmpy = xloc*tmpy;
  const rajaperf::Real_type tmpxz    = tmpx*tmpz;
  const rajaperf::Real_type tmpyz    = tmpy*tmpz;
  const rajaperf::Real_type xzloc    = xloc*zloc;
  const rajaperf::Real_type yzloc    = yloc*zloc;
  const rajaperf::Real_type tmpyzloc = tmpy*zloc;
  const rajaperf::Real_type tmpxzloc = tmpx*zloc;
  const rajaperf::Real_type yloctmpz = yloc*tmpz;
  const rajaperf::Real_type xloctmpz = xloc*tmpz;

  edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);
  edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);
  edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);
}

constexpr rajaperf::Int_type flops_transform_basis(int basis_size)
{
  return 3*5*basis_size;
}

RAJA_HOST_DEVICE
constexpr void transform_edge_basis(
  const rajaperf::Real_type jinvxx,
  const rajaperf::Real_type jinvxy,
  const rajaperf::Real_type jinvxz,
  const rajaperf::Real_type jinvyx,
  const rajaperf::Real_type jinvyy,
  const rajaperf::Real_type jinvyz,
  const rajaperf::Real_type jinvzx,
  const rajaperf::Real_type jinvzy,
  const rajaperf::Real_type jinvzz,
  rajaperf::Real_type (&basisx)[EB],
  rajaperf::Real_type (&basisy)[EB],
  rajaperf::Real_type (&basisz)[EB],
  rajaperf::Real_type (&tbasisx)[EB],
  rajaperf::Real_type (&tbasisy)[EB],
  rajaperf::Real_type (&tbasisz)[EB])
{
  // Transform is:  w_i <- J^{-1} w_i
  transform_basis(
    jinvxx, jinvxy, jinvxz,
    jinvyx, jinvyy, jinvyz,
    jinvzx, jinvzy, jinvzz,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}

RAJA_HOST_DEVICE
constexpr void transform_curl_edge_basis(
  const rajaperf::Real_type jxx,
  const rajaperf::Real_type jxy,
  const rajaperf::Real_type jxz,
  const rajaperf::Real_type jyx,
  const rajaperf::Real_type jyy,
  const rajaperf::Real_type jyz,
  const rajaperf::Real_type jzx,
  const rajaperf::Real_type jzy,
  const rajaperf::Real_type jzz,
  const rajaperf::Real_type invdetj,
  rajaperf::Real_type (&basisx)[EB],
  rajaperf::Real_type (&basisy)[EB],
  rajaperf::Real_type (&basisz)[EB],
  rajaperf::Real_type (&tbasisx)[EB],
  rajaperf::Real_type (&tbasisy)[EB],
  rajaperf::Real_type (&tbasisz)[EB])
{
  // Transform is:  Curl(w_i) <- (1/|J|)J^{T} Curl(w_i)
  transform_basis(
    jxx*invdetj, jyx*invdetj, jzx*invdetj,
    jxy*invdetj, jyy*invdetj, jzy*invdetj,
    jxz*invdetj, jyz*invdetj, jzz*invdetj,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}

//-----------------------------------------
// Face basis
//-----------------------------------------
RAJA_HOST_DEVICE
constexpr void face_basis_x(
  rajaperf::Real_type (&basisx)[FB],
  const rajaperf::Real_type tmpx,
  const rajaperf::Real_type xpt)
{
  basisx[0] = tmpx;
  basisx[1] = xpt;
  basisx[2] = 0.0;
  basisx[3] = 0.0;
  basisx[4] = 0.0;
  basisx[5] = 0.0;
}

RAJA_HOST_DEVICE
constexpr void face_basis_y(
  rajaperf::Real_type (&basisy)[FB],
  const rajaperf::Real_type tmpy,
  const rajaperf::Real_type ypt)
{
  basisy[0] = 0.0;
  basisy[1] = 0.0;
  basisy[2] = tmpy;
  basisy[3] = ypt;
  basisy[4] = 0.0;
  basisy[5] = 0.0;
}

RAJA_HOST_DEVICE
constexpr void face_basis_z(
  rajaperf::Real_type (&basisz)[FB],
  const rajaperf::Real_type tmpz,
  const rajaperf::Real_type zpt)
{
  basisz[0] = 0.0;
  basisz[1] = 0.0;
  basisz[2] = 0.0;
  basisz[3] = 0.0;
  basisz[4] = tmpz;
  basisz[5] = zpt;
}

RAJA_HOST_DEVICE
constexpr void transform_face_basis(
  const rajaperf::Real_type jxx,
  const rajaperf::Real_type jxy,
  const rajaperf::Real_type jxz,
  const rajaperf::Real_type jyx,
  const rajaperf::Real_type jyy,
  const rajaperf::Real_type jyz,
  const rajaperf::Real_type jzx,
  const rajaperf::Real_type jzy,
  const rajaperf::Real_type jzz,
  const rajaperf::Real_type invdetj,
  rajaperf::Real_type (&basisx)[FB],
  rajaperf::Real_type (&basisy)[FB],
  rajaperf::Real_type (&basisz)[FB],
  rajaperf::Real_type (&tbasisx)[FB],
  rajaperf::Real_type (&tbasisy)[FB],
  rajaperf::Real_type (&tbasisz)[FB])
{
  // Transform is:  f_i <- (1/|J|)J^{T} f_i
  transform_basis(
    jxx*invdetj, jyx*invdetj, jzx*invdetj,
    jxy*invdetj, jyy*invdetj, jzy*invdetj,
    jxz*invdetj, jyz*invdetj, jzz*invdetj,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}

#endif  // closing endif for header file include guard
