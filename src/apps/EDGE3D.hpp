//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
/*
NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

for (Index_type i = ibegin ; i < iend ; ++i ) {
  const Real_type qpts_1d[NQ_1D] = { 1. };
  const Real_type wgts_1d[NQ_1D] = { 1. };
  Compute cell centered Jacobian
  const Real_type jxx_cc =
    (x1[i] - x0[i])*0.25 +
    (x3[i] - x2[i])*0.25 +
    (x4[i] - x5[i])*0.25 +
    (x6[i] - x7[i])*0.25;
  const Real_type jxy_cc =
    (y1[i] - y0[i])*0.25 +
    (y3[i] - y2[i])*0.25 +
    (y4[i] - y5[i])*0.25 +
    (y6[i] - y7[i])*0.25;
  const Real_type jxz_cc =
    (z1[i] - z0[i])*0.25 +
    (z3[i] - z2[i])*0.25 +
    (z4[i] - z5[i])*0.25 +
    (z6[i] - z7[i])*0.25;
  const Real_type jyx_cc =
    (x3[i] - x0[i])*0.25 +
    (x1[i] - x2[i])*0.25 +
    (x4[i] - x7[i])*0.25 +
    (x6[i] - x5[i])*0.25;
  const Real_type jyy_cc =
    (y3[i] - y0[i])*0.25 +
    (y1[i] - y2[i])*0.25 +
    (y4[i] - y7[i])*0.25 +
    (y6[i] - y5[i])*0.25;
  const Real_type jyz_cc =
    (z3[i] - z0[i])*0.25 +
    (z1[i] - z2[i])*0.25 +
    (z4[i] - z7[i])*0.25 +
    (z6[i] - z5[i])*0.25;
  const Real_type jzx_cc =
    (x4[i] - x0[i])*0.25 +
    (x1[i] - x5[i])*0.25 +
    (x6[i] - x2[i])*0.25 +
    (x3[i] - x7[i])*0.25;
  const Real_type jzy_cc =
    (y4[i] - y0[i])*0.25 +
    (y1[i] - y5[i])*0.25 +
    (y6[i] - y2[i])*0.25 +
    (y3[i] - y7[i])*0.25;
  const Real_type jzz_cc =
    (z4[i] - z0[i])*0.25 +
    (z1[i] - z5[i])*0.25 +
    (z6[i] - z2[i])*0.25 +
    (z3[i] - z7[i])*0.25;
  Compute cell centered Jacobian determinant
  const Real_type detj_cc =
    jxy_cc*jyz_cc*jzx_cc - jxz_cc*jyy_cc*jzx_cc + jxz_cc*jyx_cc*jzy_cc -
    jxx_cc*jyz_cc*jzy_cc - jxy_cc*jyx_cc*jzz_cc + jxx_cc*jyy_cc*jzz_cc;
  Real_type edge_matrix[NB*NB];
  Initialize the matrix
  for (int m = 0; m < NB; m++) {
    for (int p = 0; p < NB; p++) {
      edge_matrix[m*NB + p] = 0.0;
    }
  }
  Compute values at each quadrature point
  for ( int qi = 0; qi < NQ_1D; qi++ ) {
    const Real_type xloc = qpts_1d[qi];
    const Real_type tmpx = 1. - xloc;
    Uses fake values, basis = 1
    const Real_type dbasisx[NB] = {1.0};
    for ( int qj = 0; qj < NQ_1D; qj++ ) {
      const Real_type yloc = qpts_1d[qj];
      const Real_type wgtxy = wgts_1d[qi]*wgts_1d[qj];
      const Real_type tmpy = 1. - yloc;
      const Real_type tmpxy    = tmpx*tmpy;
      const Real_type xyloc    = xloc*yloc;
      const Real_type tmpxyloc = tmpx*yloc;
      const Real_type xloctmpy = xloc*tmpy;
      const Real_type jzx =
        (x4[i] - x0[i])*tmpxy    +
        (x5[i] - x1[i])*xloctmpy +
        (x6[i] - x2[i])*xyloc    +
        (x7[i] - x3[i])*tmpxyloc;
      const Real_type jzy =
        (y4[i] - y0[i])*tmpxy    +
        (y5[i] - y1[i])*xloctmpy +
        (y6[i] - y2[i])*xyloc    +
        (y7[i] - y3[i])*tmpxyloc;
      const Real_type jzz =
        (z4[i] - z0[i])*tmpxy    +
        (z5[i] - z1[i])*xloctmpy +
        (z6[i] - z2[i])*xyloc    +
        (z7[i] - z3[i])*tmpxyloc;
      const Real_type basisz[NB] = {1.0};
      const Real_type dbasisy[NB] = {1.0};
      Differeniate basis with respect to z at this quadrature point
      for ( int qk = 0; qk < NQ_1D; qk++ ) {
        const Real_type zloc = qpts_1d[qk];
        const Real_type wgts = wgtxy*wgts_1d[qk];
        const Real_type tmpz = 1. - zloc;
        const Real_type tmpxz    = tmpx*tmpz;
        const Real_type tmpyz    = tmpy*tmpz;
        const Real_type xzloc    = xloc*zloc;
        const Real_type yzloc    = yloc*zloc;
        const Real_type tmpyzloc = tmpy*zloc;
        const Real_type tmpxzloc = tmpx*zloc;
        const Real_type yloctmpz = yloc*tmpz;
        const Real_type xloctmpz = xloc*tmpz;
        const Real_type jxx =
          (x1[i] - x0[i])*tmpyz   +
          (x2[i] - x3[i])*yloctmpz +
          (x5[i] - x4[i])*tmpyzloc +
          (x6[i] - x7[i])*yzloc;
        const Real_type jxy =
          (y1[i] - y0[i])*tmpyz    +
          (y2[i] - y3[i])*yloctmpz +
          (y5[i] - y4[i])*tmpyzloc +
          (y6[i] - y7[i])*yzloc;
        const Real_type jxz =
          (z1[i] - z0[i])*tmpyz   +
          (z2[i] - z3[i])*yloctmpz +
          (z5[i] - z4[i])*tmpyzloc +
          (z6[i] - z7[i])*yzloc;
        const Real_type jyx =
          (x3[i] - x0[i])*tmpxz    +
          (x2[i] - x1[i])*xloctmpz +
          (x7[i] - x4[i])*tmpxzloc +
          (x6[i] - x5[i])*xzloc;
        const Real_type jyy =
          (y3[i] - y0[i])*tmpxz    +
          (y2[i] - y1[i])*xloctmpz +
          (y7[i] - y4[i])*tmpxzloc +
          (y6[i] - y5[i])*xzloc;
        const Real_type jyz =
          (z3[i] - z0[i])*tmpxz    +
          (z2[i] - z1[i])*xloctmpz +
          (z7[i] - z4[i])*tmpxzloc +
          (z6[i] - z5[i])*xzloc;
        const Real_type basisx[NB] = {1.0};
        const Real_type basisy[NB] = {1.0};
        const Real_type dbasisz[NB] = {1.0};
        Compute determinant of Jacobian matrix at this quadrature point
        const Real_type detj_pre =
          jxy*jyz*jzx - jxz*jyy*jzx + jxz*jyx*jzy -
          jxx*jyz*jzy - jxy*jyx*jzz + jxx*jyy*jzz;
        "Bad Zone" detection algorithm:
        const Real_type detj_tol = 0.;
        const Real_type detj = (std::fabs( detj_pre/detj_cc ) < detj_tol) ? detj_cc : detj_pre ;
        const Real_type abs_detj = std::fabs(detj);
        const Real_type abs_detj_pre = std::fabs(detj_pre);
        const Real_type inv_detj_pre = 1.0/abs_detj_pre;
        const Real_type inv_detj = inv_detj_pre;
        const Real_type detjwgts = wgts*abs_detj;
        const Real_type jinvxx =  (jyy*jzz - jyz*jzy)*inv_detj;
        const Real_type jinvxy =  (jxz*jzy - jxy*jzz)*inv_detj;
        const Real_type jinvxz =  (jxy*jyz - jxz*jyy)*inv_detj;
        const Real_type jinvyx =  (jyz*jzx - jyx*jzz)*inv_detj;
        const Real_type jinvyy =  (jxx*jzz - jxz*jzx)*inv_detj;
        const Real_type jinvyz =  (jxz*jyx - jxx*jyz)*inv_detj;
        const Real_type jinvzx =  (jyx*jzy - jyy*jzx)*inv_detj;
        const Real_type jinvzy =  (jxy*jzx - jxx*jzy)*inv_detj;
        const Real_type jinvzz =  (jxx*jyy - jxy*jyx)*inv_detj;
        Real_type tdbasisx[NB];
        Real_type tdbasisy[NB];
        Real_type tdbasisz[NB];
        Compute transformed basis function gradients
        for (int m = 0; m < NB; m++) {
          Transform is:  Grad(w_i) <- J^{-1} Grad(w_i)
          tdbasisx[m] = jinvxx*dbasisx[m] + jinvxy*dbasisy[m] + jinvxz*dbasisz[m];
          tdbasisy[m] = jinvyx*dbasisx[m] + jinvyy*dbasisy[m] + jinvyz*dbasisz[m];
          tdbasisz[m] = jinvzx*dbasisx[m] + jinvzy*dbasisy[m] + jinvzz*dbasisz[m];
        }
        Real_type tbasisx[NB];
        Real_type tbasisy[NB];
        Real_type tbasisz[NB];
        for (int m = 0; m < NB; m++) {
          Transform is:  w_i <- J^{-1} w_i
          tbasisx[m] = jinvxx*basisx[m] + jinvxy*basisy[m] + jinvxz*basisz[m];
          tbasisy[m] = jinvyx*basisx[m] + jinvyy*basisy[m] + jinvyz*basisz[m];
          tbasisz[m] = jinvzx*basisx[m] + jinvzy*basisy[m] + jinvzz*basisz[m];
          Transform is:  Curl(w_i) <- (1/|J|)J^{T} Curl(w_i)
          tdbasisx[m] = jxx*dbasisx[m] + jyx*dbasisy[m] + jzx*dbasisz[m];
          tdbasisy[m] = jxy*dbasisx[m] + jyy*dbasisy[m] + jzy*dbasisz[m];
          tdbasisz[m] = jxz*dbasisx[m] + jyz*dbasisy[m] + jzz*dbasisz[m];
          tdbasisx[m] *= inv_detj_pre;
          tdbasisy[m] *= inv_detj_pre;
          tdbasisz[m] *= inv_detj_pre;
        }
        Compute the local mass plus stiffness matrix
        for (int m = 0; m < NB; m++) {
          const Real_type txm = tbasisx[m];
          const Real_type tym = tbasisy[m];
          const Real_type tzm = tbasisz[m];
          const Real_type dtxm = tdbasisx[m];
          const Real_type dtym = tdbasisy[m];
          const Real_type dtzm = tdbasisz[m];
          Compute the upper triangular portion
          for (int p = m; p < NB; p++) {
            inner product: <w_i, w_j>
            const Real_type txp = tbasisx[p];
            const Real_type typ = tbasisy[p];
            const Real_type tzp = tbasisz[p];
            const Real_type Mtemp = detjwgts*(txm*txp + tym*typ + tzm*tzp);
            inner product: <Curl(w_i), Curl(w_j)>
            const Real_type dtxp = tdbasisx[p];
            const Real_type dtyp = tdbasisy[p];
            const Real_type dtzp = tdbasisz[p];
            const Real_type Stemp = detjwgts*(dtxm*dtxp + dtym*dtyp + dtzm*dtzp);
            Add the entries to the matrix
            const Real_type x = Mtemp + Stemp;
            edge_matrix[p*NB + m] = x;
            edge_matrix[m*NB + p] = x;
          }
        }
      }
    }
  }
  sum[i] = 0.0;
  for (int m = 0; m < NB; m++) {
    Real_type check = 0.0;
    for (int p = 0; p < NB; p++) {
      check += edge_matrix[m*NB + p];
    }
    sum[i] += check;
  }
}
*/
///

#ifndef RAJAPerf_Apps_EDGE3D_HPP
#define RAJAPerf_Apps_EDGE3D_HPP

#define EDGE3D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
  Real_ptr sum = m_sum; \
\
  Real_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  Real_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  Real_ptr z0,z1,z2,z3,z4,z5,z6,z7 ;

//Number of Dofs/Qpts in 1D
// NB = 12 is the need value
// NB can vary to probe kernel performance
#define NB 12
#define NQ_1D 2

#define EDGE3D_BODY \
  const Real_type qpts_1d[NQ_1D] = { 1. };\
  const Real_type wgts_1d[NQ_1D] = { 1. };\
  /* Compute cell centered Jacobian */\
  const Real_type jxx_cc =\
    (x1[i] - x0[i])*0.25 +\
    (x3[i] - x2[i])*0.25 +\
    (x4[i] - x5[i])*0.25 +\
    (x6[i] - x7[i])*0.25;\
  const Real_type jxy_cc =\
    (y1[i] - y0[i])*0.25 +\
    (y3[i] - y2[i])*0.25 +\
    (y4[i] - y5[i])*0.25 +\
    (y6[i] - y7[i])*0.25;\
  const Real_type jxz_cc =\
    (z1[i] - z0[i])*0.25 +\
    (z3[i] - z2[i])*0.25 +\
    (z4[i] - z5[i])*0.25 +\
    (z6[i] - z7[i])*0.25;\
  const Real_type jyx_cc =\
    (x3[i] - x0[i])*0.25 +\
    (x1[i] - x2[i])*0.25 +\
    (x4[i] - x7[i])*0.25 +\
    (x6[i] - x5[i])*0.25;\
  const Real_type jyy_cc =\
    (y3[i] - y0[i])*0.25 +\
    (y1[i] - y2[i])*0.25 +\
    (y4[i] - y7[i])*0.25 +\
    (y6[i] - y5[i])*0.25;\
  const Real_type jyz_cc =\
    (z3[i] - z0[i])*0.25 +\
    (z1[i] - z2[i])*0.25 +\
    (z4[i] - z7[i])*0.25 +\
    (z6[i] - z5[i])*0.25;\
  const Real_type jzx_cc =\
    (x4[i] - x0[i])*0.25 +\
    (x1[i] - x5[i])*0.25 +\
    (x6[i] - x2[i])*0.25 +\
    (x3[i] - x7[i])*0.25;\
  const Real_type jzy_cc =\
    (y4[i] - y0[i])*0.25 +\
    (y1[i] - y5[i])*0.25 +\
    (y6[i] - y2[i])*0.25 +\
    (y3[i] - y7[i])*0.25;\
  const Real_type jzz_cc =\
    (z4[i] - z0[i])*0.25 +\
    (z1[i] - z5[i])*0.25 +\
    (z6[i] - z2[i])*0.25 +\
    (z3[i] - z7[i])*0.25;\
  /* Compute cell centered Jacobian determinant */ \
  const Real_type detj_cc = \
    jxy_cc*jyz_cc*jzx_cc - jxz_cc*jyy_cc*jzx_cc + jxz_cc*jyx_cc*jzy_cc -\
    jxx_cc*jyz_cc*jzy_cc - jxy_cc*jyx_cc*jzz_cc + jxx_cc*jyy_cc*jzz_cc;\
  Real_type edge_matrix[NB*NB];\
  /* Initialize the matrix */\
  for (int m = 0; m < NB; m++) {\
    for (int p = 0; p < NB; p++) {\
      edge_matrix[m*NB + p] = 0.0;\
    }\
  }\
  /* Compute values at each quadrature point*/ \
  for ( int qi = 0; qi < NQ_1D; qi++ ) {\
    const Real_type xloc = qpts_1d[qi];\
    const Real_type tmpx = 1. - xloc;\
    /* Uses fake values, basis = 1 */ \
    const Real_type dbasisx[NB] = {1.0}; \
    for ( int qj = 0; qj < NQ_1D; qj++ ) {\
      const Real_type yloc = qpts_1d[qj];\
      const Real_type wgtxy = wgts_1d[qi]*wgts_1d[qj];\
      const Real_type tmpy = 1. - yloc;\
      const Real_type tmpxy    = tmpx*tmpy;\
      const Real_type xyloc    = xloc*yloc;\
      const Real_type tmpxyloc = tmpx*yloc;\
      const Real_type xloctmpy = xloc*tmpy;\
      const Real_type jzx =\
        (x4[i] - x0[i])*tmpxy    +\
        (x5[i] - x1[i])*xloctmpy +\
        (x6[i] - x2[i])*xyloc    +\
        (x7[i] - x3[i])*tmpxyloc;\
      const Real_type jzy =\
        (y4[i] - y0[i])*tmpxy    +\
        (y5[i] - y1[i])*xloctmpy +\
        (y6[i] - y2[i])*xyloc    +\
        (y7[i] - y3[i])*tmpxyloc;\
      const Real_type jzz =\
        (z4[i] - z0[i])*tmpxy    +\
        (z5[i] - z1[i])*xloctmpy +\
        (z6[i] - z2[i])*xyloc    +\
        (z7[i] - z3[i])*tmpxyloc;\
      const Real_type basisz[NB] = {1.0}; \
      const Real_type dbasisy[NB] = {1.0}; \
      /* Differeniate basis with respect to z at this quadrature point */\
      for ( int qk = 0; qk < NQ_1D; qk++ ) {\
        const Real_type zloc = qpts_1d[qk];\
        const Real_type wgts = wgtxy*wgts_1d[qk];\
        const Real_type tmpz = 1. - zloc;\
        const Real_type tmpxz    = tmpx*tmpz;\
        const Real_type tmpyz    = tmpy*tmpz;\
        const Real_type xzloc    = xloc*zloc;\
        const Real_type yzloc    = yloc*zloc;\
        const Real_type tmpyzloc = tmpy*zloc;\
        const Real_type tmpxzloc = tmpx*zloc;\
        const Real_type yloctmpz = yloc*tmpz;\
        const Real_type xloctmpz = xloc*tmpz;\
        const Real_type jxx =\
          (x1[i] - x0[i])*tmpyz   +\
          (x2[i] - x3[i])*yloctmpz +\
          (x5[i] - x4[i])*tmpyzloc +\
          (x6[i] - x7[i])*yzloc;\
        const Real_type jxy =\
          (y1[i] - y0[i])*tmpyz    +\
          (y2[i] - y3[i])*yloctmpz +\
          (y5[i] - y4[i])*tmpyzloc +\
          (y6[i] - y7[i])*yzloc;\
        const Real_type jxz =\
          (z1[i] - z0[i])*tmpyz   +\
          (z2[i] - z3[i])*yloctmpz +\
          (z5[i] - z4[i])*tmpyzloc +\
          (z6[i] - z7[i])*yzloc;\
        const Real_type jyx =\
          (x3[i] - x0[i])*tmpxz    +\
          (x2[i] - x1[i])*xloctmpz +\
          (x7[i] - x4[i])*tmpxzloc +\
          (x6[i] - x5[i])*xzloc;\
        const Real_type jyy =\
          (y3[i] - y0[i])*tmpxz    +\
          (y2[i] - y1[i])*xloctmpz +\
          (y7[i] - y4[i])*tmpxzloc +\
          (y6[i] - y5[i])*xzloc;\
        const Real_type jyz =\
          (z3[i] - z0[i])*tmpxz    +\
          (z2[i] - z1[i])*xloctmpz +\
          (z7[i] - z4[i])*tmpxzloc +\
          (z6[i] - z5[i])*xzloc;\
        const Real_type basisx[NB] = {1.0}; \
        const Real_type basisy[NB] = {1.0}; \
        const Real_type dbasisz[NB] = {1.0}; \
        /* Compute determinant of Jacobian matrix at this quadrature point*/ \
        const Real_type detj_pre =\
          jxy*jyz*jzx - jxz*jyy*jzx + jxz*jyx*jzy -\
          jxx*jyz*jzy - jxy*jyx*jzz + jxx*jyy*jzz;\
        /* "Bad Zone" detection algorithm: */ \
        const Real_type detj_tol = 0.;\
        const Real_type detj = (std::fabs( detj_pre/detj_cc ) < detj_tol) ? detj_cc : detj_pre ;\
        const Real_type abs_detj = std::fabs(detj);\
        const Real_type abs_detj_pre = std::fabs(detj_pre);\
        const Real_type inv_detj_pre = 1.0/abs_detj_pre;\
        const Real_type inv_detj = inv_detj_pre;\
        const Real_type detjwgts = wgts*abs_detj;\
        const Real_type jinvxx =  (jyy*jzz - jyz*jzy)*inv_detj;\
        const Real_type jinvxy =  (jxz*jzy - jxy*jzz)*inv_detj;\
        const Real_type jinvxz =  (jxy*jyz - jxz*jyy)*inv_detj;\
        const Real_type jinvyx =  (jyz*jzx - jyx*jzz)*inv_detj;\
        const Real_type jinvyy =  (jxx*jzz - jxz*jzx)*inv_detj;\
        const Real_type jinvyz =  (jxz*jyx - jxx*jyz)*inv_detj;\
        const Real_type jinvzx =  (jyx*jzy - jyy*jzx)*inv_detj;\
        const Real_type jinvzy =  (jxy*jzx - jxx*jzy)*inv_detj;\
        const Real_type jinvzz =  (jxx*jyy - jxy*jyx)*inv_detj;\
        Real_type tdbasisx[NB];\
        Real_type tdbasisy[NB];\
        Real_type tdbasisz[NB];\
        /* Compute transformed basis function gradients */\
        for (int m = 0; m < NB; m++) {\
          /* Transform is:  Grad(w_i) <- J^{-1} Grad(w_i) */\
          tdbasisx[m] = jinvxx*dbasisx[m] + jinvxy*dbasisy[m] + jinvxz*dbasisz[m];\
          tdbasisy[m] = jinvyx*dbasisx[m] + jinvyy*dbasisy[m] + jinvyz*dbasisz[m];\
          tdbasisz[m] = jinvzx*dbasisx[m] + jinvzy*dbasisy[m] + jinvzz*dbasisz[m];\
        }\
        Real_type tbasisx[NB];\
        Real_type tbasisy[NB];\
        Real_type tbasisz[NB];\
        for (int m = 0; m < NB; m++) {\
          /* Transform is:  w_i <- J^{-1} w_i */\
          tbasisx[m] = jinvxx*basisx[m] + jinvxy*basisy[m] + jinvxz*basisz[m];\
          tbasisy[m] = jinvyx*basisx[m] + jinvyy*basisy[m] + jinvyz*basisz[m];\
          tbasisz[m] = jinvzx*basisx[m] + jinvzy*basisy[m] + jinvzz*basisz[m];\
          /* Transform is:  Curl(w_i) <- (1/|J|)J^{T} Curl(w_i) */\
          tdbasisx[m] = jxx*dbasisx[m] + jyx*dbasisy[m] + jzx*dbasisz[m];\
          tdbasisy[m] = jxy*dbasisx[m] + jyy*dbasisy[m] + jzy*dbasisz[m];\
          tdbasisz[m] = jxz*dbasisx[m] + jyz*dbasisy[m] + jzz*dbasisz[m];\
          tdbasisx[m] *= inv_detj_pre;\
          tdbasisy[m] *= inv_detj_pre;\
          tdbasisz[m] *= inv_detj_pre;\
        }\
        /* Compute the local mass plus stiffness matrix */\
        for (int m = 0; m < NB; m++) {\
          const Real_type txm = tbasisx[m];\
          const Real_type tym = tbasisy[m];\
          const Real_type tzm = tbasisz[m];\
          const Real_type dtxm = tdbasisx[m];\
          const Real_type dtym = tdbasisy[m];\
          const Real_type dtzm = tdbasisz[m];\
          /* Compute the upper triangular portion */\
          for (int p = m; p < NB; p++) {\
            /* inner product: <w_i, w_j> */\
            const Real_type txp = tbasisx[p];\
            const Real_type typ = tbasisy[p];\
            const Real_type tzp = tbasisz[p];\
            const Real_type Mtemp = detjwgts*(txm*txp + tym*typ + tzm*tzp);\
            /* inner product: <Curl(w_i), Curl(w_j)> */\
            const Real_type dtxp = tdbasisx[p];\
            const Real_type dtyp = tdbasisy[p];\
            const Real_type dtzp = tdbasisz[p];\
            const Real_type Stemp = detjwgts*(dtxm*dtxp + dtym*dtyp + dtzm*dtzp);\
            /* Add the entries to the matrix*/\
            const Real_type x = Mtemp + Stemp;\
            edge_matrix[p*NB + m] = x;\
            edge_matrix[m*NB + p] = x;\
          }\
        }\
      }\
    }\
  }\
  sum[i] = 0.0;\
  for (int m = 0; m < NB; m++) {\
    Real_type check = 0.0;\
    for (int p = 0; p < NB; p++) {\
      check += edge_matrix[m*NB + p];\
    }\
    sum[i] += check;\
  }

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
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

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
