
#pragma once

#define NB 8
#define EB 12
#define FB 6
#define MAX_QUAD_ORDER 5

constexpr double ptiny = 1.0e-50;

//
// Common FEM functions
//

__host__ __device__
inline void LinAlg_qrule_Lobatto(
  int    order,
  double *qpts1D,
  double *wgts1D)
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

__host__ __device__
inline void LinAlg_qrule_Legendre(
  int    order,
  double *qpts1D,
  double *wgts1D)
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

__host__ __device__
inline void get_quadrature_rule(
  const int    quad_type,
  const int    quad_order,
  double       (&qpts_1d)[MAX_QUAD_ORDER],
  double       (&wgts_1d)[MAX_QUAD_ORDER])
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

__host__ __device__
constexpr double compute_detj(
  const double jxx,
  const double jxy,
  const double jxz,
  const double jyx,
  const double jyy,
  const double jyz,
  const double jzx,
  const double jzy,
  const double jzz)
{
  return
    jxy*jyz*jzx - jxz*jyy*jzx + jxz*jyx*jzy -
    jxx*jyz*jzy - jxy*jyx*jzz + jxx*jyy*jzz;
}

template<int M>
__host__ __device__
constexpr void transform_basis(
  const double txx,
  const double txy,
  const double txz,
  const double tyx,
  const double tyy,
  const double tyz,
  const double tzx,
  const double tzy,
  const double tzz,
  const double (&basis_x)[M],
  const double (&basis_y)[M],
  const double (&basis_z)[M],
  double (&tbasis_x)[M],
  double (&tbasis_y)[M],
  double (&tbasis_z)[M])
{
  // Compute transformed basis function gradients
  for (int m = 0; m < M; m++)
  {
    tbasis_x[m] = txx*basis_x[m] + txy*basis_y[m] + txz*basis_z[m];
    tbasis_y[m] = tyx*basis_x[m] + tyy*basis_y[m] + tyz*basis_z[m];
    tbasis_z[m] = tzx*basis_x[m] + tzy*basis_y[m] + tzz*basis_z[m];
  }
}

template<int M, int P>
__host__ __device__
constexpr void inner_product(
  const double weight,
  const double (&basis_1_x)[M],
  const double (&basis_1_y)[M],
  const double (&basis_1_z)[M],
  const double (&basis_2_x)[P],
  const double (&basis_2_y)[P],
  const double (&basis_2_z)[P],
  double (&matrix)[P][M],
  const bool is_symmetric)
{
  // inner product is <basis_2, basis_1>
  for (int p = 0; p < P; p++) {

    const double txi = basis_2_x[p];
    const double tyi = basis_2_y[p];
    const double tzi = basis_2_z[p];

    const int m0 = (is_symmetric && (M == P)) ? p : 0;

    for (int m = m0; m < M; m++) {

      const double txj = basis_1_x[m];
      const double tyj = basis_1_y[m];
      const double tzj = basis_1_z[m];

      matrix[p][m] += weight*(txi*txj + tyi*tyj + tzi*tzj);

      if(is_symmetric && (M == P) && (m > m0))
      {
        matrix[m][p] = matrix[p][m];
      }
    }
  }
}

__host__ __device__
inline void bad_zone_algorithm(
  const double detj_unfixed,
  const double detj_cc,
  const double detj_tol,
  double& detj,
  double& abs_detj,
  double& inv_detj)
{
  detj = (fabs( detj_unfixed/detj_cc ) < detj_tol) ?
                               detj_cc : detj_unfixed ;
  abs_detj = fabs(detj);

  // Note that this uses a potentially negative detj

  inv_detj = 1.0/(detj + ptiny);
}

__host__ __device__
inline void jacobian_inv(
  const double jxx,
  const double jxy,
  const double jxz,
  const double jyx,
  const double jyy,
  const double jyz,
  const double jzx,
  const double jzy,
  const double jzz,
  const double detj_cc,
  const double detj_tol,
  double &jinvxx,
  double &jinvxy,
  double &jinvxz,
  double &jinvyx,
  double &jinvyy,
  double &jinvyz,
  double &jinvzx,
  double &jinvzy,
  double &jinvzz,
  double &detj_unfixed,
  double &detj,
  double &abs_detj,
  double &inv_detj)
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

__host__ __device__
constexpr double Jzx(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxy,
  const double  xloctmpy,
  const double  xyloc,
  const double  tmpxyloc)
{
  return
    (x[4] - x[0])*tmpxy    +
    (x[5] - x[1])*xloctmpy +
    (x[6] - x[2])*xyloc    +
    (x[7] - x[3])*tmpxyloc;
}

__host__ __device__
constexpr double Jzy(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxy,
  const double  xloctmpy,
  const double  xyloc,
  const double  tmpxyloc)
{
  return
    (y[4] - y[0])*tmpxy    +
    (y[5] - y[1])*xloctmpy +
    (y[6] - y[2])*xyloc    +
    (y[7] - y[3])*tmpxyloc;
}

__host__ __device__
constexpr double Jzz(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxy,
  const double  xloctmpy,
  const double  xyloc,
  const double  tmpxyloc)
{
  return
    (z[4] - z[0])*tmpxy    +
    (z[5] - z[1])*xloctmpy +
    (z[6] - z[2])*xyloc    +
    (z[7] - z[3])*tmpxyloc;
}

__host__ __device__
constexpr double Jxx(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpyz,
  const double  yloctmpz,
  const double  tmpyzloc,
  const double  yzloc)
{
  return
    (x[1] - x[0])*tmpyz   +
    (x[2] - x[3])*yloctmpz +
    (x[5] - x[4])*tmpyzloc +
    (x[6] - x[7])*yzloc;
}

__host__ __device__
constexpr double Jxy(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpyz,
  const double  yloctmpz,
  const double  tmpyzloc,
  const double  yzloc)
{
  return
    (y[1] - y[0])*tmpyz    +
    (y[2] - y[3])*yloctmpz +
    (y[5] - y[4])*tmpyzloc +
    (y[6] - y[7])*yzloc;
}

__host__ __device__
constexpr double Jxz(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpyz,
  const double  yloctmpz,
  const double  tmpyzloc,
  const double  yzloc)
{
  return
    (z[1] - z[0])*tmpyz   +
    (z[2] - z[3])*yloctmpz +
    (z[5] - z[4])*tmpyzloc +
    (z[6] - z[7])*yzloc;
}

__host__ __device__
constexpr double Jyx(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxz,
  const double  xloctmpz,
  const double  tmpxzloc,
  const double  xzloc)
{
  return
    (x[3] - x[0])*tmpxz    +
    (x[2] - x[1])*xloctmpz +
    (x[7] - x[4])*tmpxzloc +
    (x[6] - x[5])*xzloc;
}

__host__ __device__
constexpr double Jyy(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxz,
  const double  xloctmpz,
  const double  tmpxzloc,
  const double  xzloc)
{
  return
    (y[3] - y[0])*tmpxz    +
    (y[2] - y[1])*xloctmpz +
    (y[7] - y[4])*tmpxzloc +
    (y[6] - y[5])*xzloc;
}

__host__ __device__
constexpr double Jyz(
  const double  (&x)[NB],
  const double  (&y)[NB],
  const double  (&z)[NB],
  const double  tmpxz,
  const double  xloctmpz,
  const double  tmpxzloc,
  const double  xzloc)
{
  return
    (z[3] - z[0])*tmpxz    +
    (z[2] - z[1])*xloctmpz +
    (z[7] - z[4])*tmpxzloc +
    (z[6] - z[5])*xzloc;
}

//-----------------------------------------
// Node basis
//-----------------------------------------
__host__ __device__
constexpr void nodebasis(
  double (&basis)[NB],
  const double tmpxy,
  const double xloctmpy,
  const double xyloc,
  const double tmpxyloc,
  const double zloc,
  const double tmpz)
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

__host__ __device__
constexpr void dnodebasis_dx(
  double (&dbasis)[NB],
  const double tmpyz,
  const double yloctmpz,
  const double tmpyzloc,
  const double yzloc)
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

__host__ __device__
constexpr void dnodebasis_dy(
  double (&dbasis)[NB],
  const double tmpxz,
  const double xloctmpz,
  const double tmpxzloc,
  const double xzloc)
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

__host__ __device__
constexpr void dnodebasis_dz(
  double (&dbasis)[NB],
  const double tmpxy,
  const double xloctmpy,
  const double xyloc,
  const double tmpxyloc)
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

__host__ __device__
constexpr void transform_node_dbasis(
  const double jinvxx,
  const double jinvxy,
  const double jinvxz,
  const double jinvyx,
  const double jinvyy,
  const double jinvyz,
  const double jinvzx,
  const double jinvzy,
  const double jinvzz,
  double (&basisx)[NB],
  double (&basisy)[NB],
  double (&basisz)[NB],
  double (&tbasisx)[NB],
  double (&tbasisy)[NB],
  double (&tbasisz)[NB])
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
__host__ __device__
constexpr void edgebasis_x(
  double (&basisx)[EB],
  const double tmpyz,
  const double yloctmpz,
  const double tmpyzloc,
  const double yzloc)
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
__host__ __device__
constexpr void edgebasis_y(
  double (&basisy)[EB],
  const double tmpxz,
  const double xloctmpz,
  const double tmpxzloc,
  const double xzloc)
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
__host__ __device__
constexpr void edgebasis_z(
  double (&basisz)[EB],
  const double tmpxy,
  const double xloctmpy,
  const double xyloc,
  const double tmpxyloc)
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
__host__ __device__
constexpr void curl_edgebasis_x(
  double (&dbasisx)[EB],
  const double tmpx,
  const double xpt)
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
__host__ __device__
constexpr void curl_edgebasis_y(
  double (&dbasisy)[EB],
  const double tmpy,
  const double ypt)
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
__host__ __device__
constexpr void curl_edgebasis_z(
  double (&dbasisz)[EB],
  const double tmpz,
  const double zpt)
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

__host__ __device__
constexpr void edgebasis(
  const double xloc,
  const double yloc,
  const double zloc,
  double (&ebasisx)[EB],
  double (&ebasisy)[EB],
  double (&ebasisz)[EB])
{
  const double tmpx = 1. - xloc;
  const double tmpy = 1. - yloc;
  const double tmpz = 1. - zloc;

  const double tmpxy    = tmpx*tmpy;
  const double xyloc    = xloc*yloc;
  const double tmpxyloc = tmpx*yloc;
  const double xloctmpy = xloc*tmpy;
  const double tmpxz    = tmpx*tmpz;
  const double tmpyz    = tmpy*tmpz;
  const double xzloc    = xloc*zloc;
  const double yzloc    = yloc*zloc;
  const double tmpyzloc = tmpy*zloc;
  const double tmpxzloc = tmpx*zloc;
  const double yloctmpz = yloc*tmpz;
  const double xloctmpz = xloc*tmpz;

  edgebasis_x(ebasisx, tmpyz, yloctmpz, tmpyzloc, yzloc);
  edgebasis_y(ebasisy, tmpxz, xloctmpz, tmpxzloc, xzloc);
  edgebasis_z(ebasisz, tmpxy, xloctmpy, xyloc, tmpxyloc);
}

__host__ __device__
constexpr void transform_edge_basis(
  const double jinvxx,
  const double jinvxy,
  const double jinvxz,
  const double jinvyx,
  const double jinvyy,
  const double jinvyz,
  const double jinvzx,
  const double jinvzy,
  const double jinvzz,
  double (&basisx)[EB],
  double (&basisy)[EB],
  double (&basisz)[EB],
  double (&tbasisx)[EB],
  double (&tbasisy)[EB],
  double (&tbasisz)[EB])
{
  // Transform is:  w_i <- J^{-1} w_i
  transform_basis(
    jinvxx, jinvxy, jinvxz,
    jinvyx, jinvyy, jinvyz,
    jinvzx, jinvzy, jinvzz,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}


__host__ __device__
constexpr void transform_curl_edge_basis(
  const double jxx,
  const double jxy,
  const double jxz,
  const double jyx,
  const double jyy,
  const double jyz,
  const double jzx,
  const double jzy,
  const double jzz,
  const double invdetj,
  double (&basisx)[EB],
  double (&basisy)[EB],
  double (&basisz)[EB],
  double (&tbasisx)[EB],
  double (&tbasisy)[EB],
  double (&tbasisz)[EB])
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
__host__ __device__
constexpr void face_basis_x(
  double (&basisx)[FB],
  const double tmpx,
  const double xpt)
{
  basisx[0] = tmpx;
  basisx[1] = xpt;
  basisx[2] = 0.0;
  basisx[3] = 0.0;
  basisx[4] = 0.0;
  basisx[5] = 0.0;
}

__host__ __device__
constexpr void face_basis_y(
  double (&basisy)[FB],
  const double tmpy,
  const double ypt)
{
  basisy[0] = 0.0;
  basisy[1] = 0.0;
  basisy[2] = tmpy;
  basisy[3] = ypt;
  basisy[4] = 0.0;
  basisy[5] = 0.0;
}

__host__ __device__
constexpr void face_basis_z(
  double (&basisz)[FB],
  const double tmpz,
  const double zpt)
{
  basisz[0] = 0.0;
  basisz[1] = 0.0;
  basisz[2] = 0.0;
  basisz[3] = 0.0;
  basisz[4] = tmpz;
  basisz[5] = zpt;
}

__host__ __device__
constexpr void transform_face_basis(
  const double jxx,
  const double jxy,
  const double jxz,
  const double jyx,
  const double jyy,
  const double jyz,
  const double jzx,
  const double jzy,
  const double jzz,
  const double invdetj,
  double (&basisx)[FB],
  double (&basisy)[FB],
  double (&basisz)[FB],
  double (&tbasisx)[FB],
  double (&tbasisy)[FB],
  double (&tbasisz)[FB])
{
  // Transform is:  f_i <- (1/|J|)J^{T} f_i
  transform_basis(
    jxx*invdetj, jyx*invdetj, jzx*invdetj,
    jxy*invdetj, jyy*invdetj, jzy*invdetj,
    jxz*invdetj, jyz*invdetj, jzz*invdetj,
    basisx, basisy, basisz,
    tbasisx, tbasisy, tbasisz);
}
