//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Apps_INTSC_HEXHEX_BODY_HPP
#define RAJAPerf_Apps_INTSC_HEXHEX_BODY_HPP

namespace rajaperf {


RAJA_HOST_DEVICE
RAJA_INLINE void clip_polygon_ge_0
    ( Real_ptr cin,   // the cut coordinate, can be xin, yin, or zin.
      Real_ptr xin, Real_ptr yin,
      Real_ptr zin, Real_ptr hin, // input coordinates
      Int_type &first, Int_type &avail, Int_ptr next )   // linked list
{
  Int_type j  = first ;

  Int_type first0 = first ;
  Int_type j1 = -1, j2 = -1 ;
  Int_type jj1 = -1, jj2 = -1 ;

  Real_type c0 = ( j >= 0 ) ? cin[j] : 0.0 ;
  Real_type c00 = c0 ;
  Real_type clast = c0 ;

  while ( j >= 0 ) {
    Int_type jj = next[j] ;
    Int_type jp = jj ;       // advancing, jp is -1 at end.
    if ( jj < 0 ) { jj = first0 ; }   // last edge of polygon

    Real_type c1 = cin[jj] ;
    if ( ( c0 >= 0 ) && ( c1 < 0 ) ) {
      j1 = j ;
      jj1 = jj ;
    }
    if ( ( c0 < 0 ) && ( c1 >= 0 ) ) {
      j2 = j ;
      jj2 = jj ;
    }
    j = jp ;
    clast = c0 ;
    c0 = c1 ;
  }

  Int_type jr1=-1, jr2=-1 ;

  if ( j1 >= 0 ) {   // Insert first crossover point

    jr1 = avail ;
    avail = next[avail] ;
    Real_type eta = ( 0.0 - cin[jj1] ) / ( cin[j1] - cin[jj1] ) ;
    xin[jr1] = xin[j1] * eta + xin[jj1] * ( 1.0 - eta ) ;
    yin[jr1] = yin[j1] * eta + yin[jj1] * ( 1.0 - eta ) ;
    zin[jr1] = zin[j1] * eta + zin[jj1] * ( 1.0 - eta ) ;
    hin[jr1] = hin[j1] * eta + hin[jj1] * ( 1.0 - eta ) ;

    jr2 = avail ;      // Insert second crossover point
    avail = next[avail] ;
    eta = ( 0.0 - cin[j2] ) / ( cin[jj2] - cin[j2] ) ;
    xin[jr2] = xin[jj2] * eta + xin[j2] * ( 1.0 - eta ) ;
    yin[jr2] = yin[jj2] * eta + yin[j2] * ( 1.0 - eta ) ;
    zin[jr2] = zin[jj2] * eta + zin[j2] * ( 1.0 - eta ) ;
    hin[jr2] = hin[jj2] * eta + hin[j2] * ( 1.0 - eta ) ;
  }

  first = -1 ;

  j = first0 ;
  while ( j >= 0 ) {   // Make removed points available.
    Int_type jp = next[j] ;
    if ( cin[j] < 0.0 ) {
      next[j] = avail ;
      avail = j ;
    } else if ( first == -1 ) {
      first = j ;        // Set first point for output polygon.
    }
    j = jp ;
  }


  if ( j1 >= 0 ) {     // Set linked list for crossover points.
    next[j1] = jr1 ;
    next[jr1] = jr2 ;
    next[jr2] = ( ( clast < 0 ) || ( c00 < 0 ) ) ? -1 : jj2 ;
  }
}



//   Simplified volume calculation, for the area under one
//   polygonal face, on the Cuda device.
//   Planar polygon.
//   Compute volume, moments between polygon and the z=0 plane.
RAJA_HOST_DEVICE
RAJA_INLINE void cuda_hex_volpolyh_1poly
    ( Real_ptr x, Real_ptr y, Real_ptr z,
      Int_type const first,
      Int_const_ptr next,
      Real_type &vv,
      Real_type &vx,
      Real_type &vy,
      Real_type &vz )
{
  if ( first < 0 ) { return ; }   // No polygon remains after clipping.

  Int_type j0 = first ;

  Real_type x0  = x[j0] ;
  Real_type y0  = y[j0] ;
  Real_type z0  = z[j0] ;

  Int_type j1 = next[j0] ;

  Real_type x1  = x[j1] ;
  Real_type y1  = y[j1] ;
  Real_type z1  = z[j1] ;
  Real_type dx1 = x1 - x0 ;
  Real_type dy1 = y1 - y0 ;

  Int_type j2 = next[j1] ;

  while ( j2 >= 0 ) {   // Vertices

    Real_type x2  = x[j2] ;
    Real_type y2  = y[j2] ;
    Real_type z2  = z[j2] ;
    Real_type dx2 = x2 - x0 ;
    Real_type dy2 = y2 - y0 ;

    Real_type area2 = (dx1 * dy2 - dx2 * dy1) ;
    Real_type v0 = ( z0 + z1 + z2 ) * area2 ;
    vv += v0 ;
    vx += v0 * ( x0 + x1 + x2 ) + area2 * ( x0 * z0 + x1 * z1 + x2 * z2 );
    vy += v0 * ( y0 + y1 + y2 ) + area2 * ( y0 * z0 + y1 * z1 + y2 * z2 );
    vz += ( z0 * z0 + z1 * z1 + z2 * z2 + z0 * z1 + z0 * z2 + z1 * z2 )
        * area2 ;

    x1=x2 ;   y1=y2 ;   z1=z2 ;    // Rotate.
    dx1=dx2 ; dy1=dy2 ;

    j2 = next[j2] ;
  }
}



RAJA_HOST_DEVICE
RAJA_INLINE void cuda_intsc_tri_tet
    ( Real_type const (&xdt)[3],    // donor triangle coordinates
      Real_type const (&ydt)[3],
      Real_type const (&zdt)[3],
      Real_type (&xtt)[4],    // target tet coordinates (modified here)
      Real_type (&ytt)[4],
      Real_type (&ztt)[4],
      Real_type &vv_thr,     // volume contribution for this triangle-tet
      Real_type &vx_thr,     // x moment contribution for this triangle-tet
      Real_type &vy_thr,     // y moment contribution for this triangle-tet
      Real_type &vz_thr )    // z moment contribution for this triangle-tet
{
  Real_type det, deti ;
  Real_type ha[9] ;      // 1 - x - y - z

  Real_type xa[9], ya[9], za[9], h2[10] ;
  Int_ptr next1 = (Int_ptr) h2 ;
  Int_ptr next  = next1 + 10 ;

  Real_type vv = 0.0, vx = 0.0, vy = 0.0, vz = 0.0 ;  // volume, moments.

  xtt[1] -= xtt[0] ;
  xtt[2] -= xtt[0] ;
  xtt[3] -= xtt[0] ;
  ytt[1] -= ytt[0] ;
  ytt[2] -= ytt[0] ;
  ytt[3] -= ytt[0] ;
  ztt[1] -= ztt[0] ;
  ztt[2] -= ztt[0] ;
  ztt[3] -= ztt[0] ;

  det =
      xtt[1]  *  ytt[2]  *  ztt[3]  - xtt[1]  *  ytt[3]  *  ztt[2]  +
      xtt[2]  *  ytt[3]  *  ztt[1]  - xtt[2]  *  ytt[1]  *  ztt[3]  +
      xtt[3]  *  ytt[1]  *  ztt[2]  - xtt[3]  *  ytt[2]  *  ztt[1]  ;
  deti = det / ( det*det + 1.0e-100 ) ;

  // Cross products.
  Real_type cyz = ytt[2] * ztt[3] - ztt[2] * ytt[3] ;
  Real_type czx = ztt[2] * xtt[3] - xtt[2] * ztt[3] ;
  Real_type cxy = xtt[2] * ytt[3] - ytt[2] * xtt[3] ;

  //   Coordinates of the facet in the transformed frame.
  xa[0] = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
      (zdt[0] - ztt[0]) * cxy ;
  xa[1] = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
      (zdt[1] - ztt[0]) * cxy ;
  xa[2] = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
      (zdt[2] - ztt[0]) * cxy ;

  cyz = ytt[3] * ztt[1] - ztt[3] * ytt[1] ;
  czx = ztt[3] * xtt[1] - xtt[3] * ztt[1] ;
  cxy = xtt[3] * ytt[1] - ytt[3] * xtt[1] ;

  ya[0] = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
      (zdt[0] - ztt[0]) * cxy ;
  ya[1] = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
      (zdt[1] - ztt[0]) * cxy ;
  ya[2] = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
      (zdt[2] - ztt[0]) * cxy ;

  cyz = ytt[1] * ztt[2] - ztt[1] * ytt[2] ;
  czx = ztt[1] * xtt[2] - xtt[1] * ztt[2] ;
  cxy = xtt[1] * ytt[2] - ytt[1] * xtt[2] ;

  za[0] = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
      (zdt[0] - ztt[0]) * cxy ;
  za[1] = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
      (zdt[1] - ztt[0]) * cxy ;
  za[2] = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
      (zdt[2] - ztt[0]) * cxy ;

  xa[0] *= deti ;    xa[1] *= deti ;    xa[2] *= deti ;
  ya[0] *= deti ;    ya[1] *= deti ;    ya[2] *= deti ;
  za[0] *= deti ;    za[1] *= deti ;    za[2] *= deti ;

  //  Clip on h2 first.
  ha[0] = 1.0 - xa[0] - ya[0] - za[0] ;
  ha[1] = 1.0 - xa[1] - ya[1] - za[1] ;
  ha[2] = 1.0 - xa[2] - ya[2] - za[2] ;
  h2[0] = 1.0 - xa[0] - ya[0] ;
  h2[1] = 1.0 - xa[1] - ya[1] ;
  h2[2] = 1.0 - xa[2] - ya[2] ;

  //  Initialize triangle and available slots.
  next[0] = 1 ;   next[1] = 2 ;   next[2] = -1 ;
  next[3] = 4 ;   next[4] = 5 ;   next[5] = 6 ;  next[6] = 7 ;
  next[7] = 8 ;   next[8] = -1 ;

  Int_type first = 0 ;
  Int_type avail = 3 ;

  clip_polygon_ge_0
      ( h2, xa, ya, za, ha, first, avail, next ) ;

  //  Clip on Cartesian faces of the unit tet.
  clip_polygon_ge_0
      ( xa, xa, ya, za, ha, first, avail, next ) ;

  clip_polygon_ge_0
      ( ya, xa, ya, za, ha, first, avail, next ) ;

  clip_polygon_ge_0
      ( za, xa, ya, za, ha, first, avail, next ) ;

  Int_type first1 = first, avail1 = avail;
  for ( Index_type k = 0 ; k < 9 ; ++k ) {
    next1[k] = next[k] ;
  }

  //  Clip on h>=0

  clip_polygon_ge_0
      ( ha, xa, ya, za, ha, first, avail, next ) ;


  cuda_hex_volpolyh_1poly( xa, ya, za, first, next, vv, vx, vy, vz ) ;


  //  In dimensionless transformed coordinates, quantity smaller
  // than machine epsilon is not significant.
  Int_type j = first1 ;
  while ( j >= 0 ) {
    ha[j] = -ha[j] - 1.0e-50 ;
    j = next1[j] ;
  }

  // Clip on h<0
  clip_polygon_ge_0
      ( ha, xa, ya, za, ha, first1, avail1, next1 ) ;

  //  project to unit tet.
  j = first1 ;
  while ( j >= 0 ) {
    za[j] = 1.0 - xa[j] - ya[j] ;
    j = next1[j] ;
  }

  cuda_hex_volpolyh_1poly( xa, ya, za, first1, next1, vv, vx, vy, vz ) ;

  //  Volume, moments of the intersection in the unit tet frame.
  vv *= 0.16666666666666667 ;
  vx *= 0.041666666666666667 ;
  vy *= 0.041666666666666667 ;
  vz *= 0.041666666666666667 ;

  //   Transform moments to the physical frame.
  vx_thr += det * (xtt[0] * vv + xtt[1] * vx + xtt[2] * vy + xtt[3] * vz);
  vy_thr += det * (ytt[0] * vv + ytt[1] * vx + ytt[2] * vy + ytt[3] * vz);
  vz_thr += det * (ztt[0] * vv + ztt[1] * vx + ztt[2] * vy + ztt[3] * vz);

  //   Transform intersection volume to the physical frame.
  vv_thr += det * vv ;
}



//  Compute the contribution of a donor triangle and a target tet
//         to intersection between hex subzones.
//   Each subzone is twelve triangular facets (six tets).
//
RAJA_HOST_DEVICE
RAJA_INLINE void hex_intsc_subz
    ( Real_const_ptr xds,    //  [24] donor subzone coords
      Real_const_ptr xts,    //  [24] target subzone coords
      Int_type const dfacet,     // which donor facet
      Int_type const ttet,       // which target tet
      Real_type &vv_thr,     // volume contribution for this triangle-tet
      Real_type &vx_thr,     // x moment contribution for this triangle-tet
      Real_type &vy_thr,     // y moment contribution for this triangle-tet
      Real_type &vz_thr )    // z moment contribution for this triangle-tet
{
  Real_const_ptr yds = xds + 8 ;
  Real_const_ptr zds = yds + 8 ;

  Real_const_ptr yts = xts + 8 ;
  Real_const_ptr zts = yts + 8 ;

  vv_thr = 0.0 ;
  vx_thr = 0.0 ;
  vy_thr = 0.0 ;
  vz_thr = 0.0 ;

  Int_type const n_dfacets = 12 ;
  Int_type const len_cycnod = n_dfacets / 2 + 1 ;

  //  coordinates of the donor triangle
  Real_type xdt[3], ydt[3], zdt[3] ;

  {
    //  cyclic nodes to form facets with node 0.
    Int_type cyc_nod[len_cycnod] = { 1, 5, 4, 6, 2, 3, 1 } ;

    // which subzone vertices form the triangular facet.
    Int_type v0, v1, v2 ;
    if ( dfacet < 6 ) {
      v0 = 0 ;
      v1 = cyc_nod[dfacet] ;
      v2 = cyc_nod[dfacet+1] ;
    } else {
      v0 = 7 ;
      v1 = cyc_nod[n_dfacets-dfacet] ;
      v2 = cyc_nod[n_dfacets-dfacet - 1] ;  // reverse order
    }

    //  Donor triangle coordinates.
    xdt[0] = xds[v0] ;    // Donor facet vertices
    xdt[1] = xds[v1] ;
    xdt[2] = xds[v2] ;
    ydt[0] = yds[v0] ;
    ydt[1] = yds[v1] ;
    ydt[2] = yds[v2] ;
    zdt[0] = zds[v0] ;
    zdt[1] = zds[v1] ;
    zdt[2] = zds[v2] ;

  }

  //   Set up the target tet and do the intersections.

  Real_type xtt[4], ytt[4], ztt[4] ;

  xtt[0] = xts[0] ;
  ytt[0] = yts[0] ;
  ztt[0] = zts[0] ;

  //  subzone vertices that form the cycle for tets.
  Int_type vert_cyc[6] = { 1, 3, 2, 6, 4, 5 } ;

  Int_type v1 = vert_cyc[ttet] ;
  xtt[1] = xts[v1] ;
  ytt[1] = yts[v1] ;
  ztt[1] = zts[v1] ;
  Int_type v2 = vert_cyc[(ttet+1)%6] ;
  xtt[2] = xts[v2] ;
  ytt[2] = yts[v2] ;
  ztt[2] = zts[v2] ;
  xtt[3] = xts[7] ;
  ytt[3] = yts[7] ;
  ztt[3] = zts[7] ;

  cuda_intsc_tri_tet
      ( xdt, ydt, zdt, xtt, ytt, ztt,
        vv_thr, vx_thr, vy_thr, vz_thr ) ;
}

}  // end namespace rajaperf


#define INTSC_HEXHEX_BODY_SEQ \
  Index_type ipair   = ith / tri_per_pair ; \
  Int_type dfacet  = ( ith / n_tsz_tets ) % n_dsz_tris ; \
  Int_type ttet    = ith % n_tsz_tets ; \
  Index_type pair_base_thr = ipair * tri_per_pair ; \
  Index_type blk_base = blk * blksize ; \
  Real_type vv_lo=0.0, vx_lo=0.0, vy_lo=0.0, vz_lo=0.0 ; \
  Real_type vv_hi=0.0, vx_hi=0.0, vy_hi=0.0, vz_hi=0.0 ; \
  if ( ipair < nisc_stage ) { \
    Real_const_ptr xds = dsubz + 24*ipair ; \
    Real_const_ptr xts = tsubz + 24*ipair ; \
    hex_intsc_subz \
        ( xds, xts, dfacet, ttet, vv_lo, vx_lo, vy_lo, vz_lo ) ; \
  } \
  if ( pair_base_thr > blk_base ) { \
    vv_hi = vv_lo ; \
    vx_hi = vx_lo ; \
    vy_hi = vy_lo ; \
    vz_hi = vz_lo ; \
    vv_lo = 0.0 ; \
    vx_lo = 0.0 ; \
    vy_lo = 0.0 ; \
    vz_lo = 0.0 ; \
  }


// thridx is threadIdx.x

#define INTSC_HEXHEX_BODY \
  INTSC_HEXHEX_BODY_SEQ \
  \
  __syncthreads() ; \
  for ( Index_type k = 1 ; k < RAJAPERF_HEXHEX_WARPSIZE ; k *= 2 ) { \
    vv_hi += RAJAPERF_HEXHEX_shfl_xor ( vv_hi, k ) ; \
    vx_hi += RAJAPERF_HEXHEX_shfl_xor ( vx_hi, k ) ; \
    vy_hi += RAJAPERF_HEXHEX_shfl_xor ( vy_hi, k ) ; \
    vz_hi += RAJAPERF_HEXHEX_shfl_xor ( vz_hi, k ) ; \
    vv_lo += RAJAPERF_HEXHEX_shfl_xor ( vv_lo, k ) ; \
    vx_lo += RAJAPERF_HEXHEX_shfl_xor ( vx_lo, k ) ; \
    vy_lo += RAJAPERF_HEXHEX_shfl_xor ( vy_lo, k ) ; \
    vz_lo += RAJAPERF_HEXHEX_shfl_xor ( vz_lo, k ) ; \
  } \
  Int_type const nwarps = blksize / RAJAPERF_HEXHEX_WARPSIZE ; \
  Int_type k = thridx / RAJAPERF_HEXHEX_WARPSIZE ; \
  if ( thridx == k*RAJAPERF_HEXHEX_WARPSIZE ) { \
    vv_reduce[ k + 0*max_warps_per_block ] = vv_lo ; \
    vv_reduce[ k + 1*max_warps_per_block ] = vx_lo ; \
    vv_reduce[ k + 2*max_warps_per_block ] = vy_lo ; \
    vv_reduce[ k + 3*max_warps_per_block ] = vz_lo ; \
    vv_reduce[ k + 4*max_warps_per_block ] = vv_hi ; \
    vv_reduce[ k + 5*max_warps_per_block ] = vx_hi ; \
    vv_reduce[ k + 6*max_warps_per_block ] = vy_hi ; \
    vv_reduce[ k + 7*max_warps_per_block ] = vz_hi ; \
  } \
  __syncthreads() ; \
  if ( thridx < max_pairs_per_block * nvals_per_pair ) { \
    for ( Index_type k = 1 ; k < nwarps ; ++k ) { \
      vv_reduce    [ max_warps_per_block*thridx     ] += \
          vv_reduce[ max_warps_per_block*thridx + k ] ;            \
    } \
    vv_int_p[thridx] = vv_reduce[ max_warps_per_block * thridx ] ; \
  }

#define INTSC_HEXHEX_SEQ(i,iend)      \
  Index_type nisc_stage = iend ; \
  Index_type blksize = default_gpu_block_size ; \
  Index_type ith = i ; \
  Index_type blk = ith / blksize ; \
  Real_ptr vv_int_p = vv_int + n_vvint_per_block * blk ; \
  if ( i == 0 ) { \
    Index_type gsize = iend / blksize ; \
    Index_type vv_int_len = n_vvint_per_block * gsize ;  \
    for ( Index_type k = 0 ; k < vv_int_len ; ++k ) { \
      vv_int_p[k] = 0.0 ; \
    } \
  } \
  INTSC_HEXHEX_BODY_SEQ ; \
  vv_int_p[0] += vv_lo ; \
  vv_int_p[1] += vx_lo ; \
  vv_int_p[2] += vy_lo ; \
  vv_int_p[3] += vz_lo ; \
  vv_int_p[4] += vv_hi ; \
  vv_int_p[5] += vx_hi ; \
  vv_int_p[6] += vy_hi ; \
  vv_int_p[7] += vz_hi ;


//  Index i is standard intersection, ipair0 = 8*i is the first
//  subzone pair for this intersection.  Initializes 32 output values
//  for the eight pairs in the first loop.
//
#define INTSC_HEXHEX_OMP(i,iend)      \
  Index_type blksize = default_gpu_block_size ; \
  Index_type nisc_stage = iend * tri_per_group ; \
  Real_ptr vv_int_p0 = vv_int + i * n_vvint_per_group ; \
  for ( Index_type j=0 ; j < n_vvint_per_group ; ++j ) { \
    vv_int_p0[ j ] = 0.0 ; \
  } \
  Index_type j0 = i * tri_per_group ; \
  for ( Index_type j = 0 ; j < tri_per_group ; ++j ) { \
    Index_type ith = j0 + j ; \
    Index_type blk = ith / blksize ; \
    INTSC_HEXHEX_BODY_SEQ ; \
    Real_ptr vv_int_p = vv_int + n_vvint_per_block * blk ; \
    vv_int_p[0] += vv_lo ; \
    vv_int_p[1] += vx_lo ; \
    vv_int_p[2] += vy_lo ; \
    vv_int_p[3] += vz_lo ; \
    vv_int_p[4] += vv_hi ; \
    vv_int_p[5] += vx_hi ; \
    vv_int_p[6] += vy_hi ; \
    vv_int_p[7] += vz_hi ; \
  }




//  This is not needed on Seq and OMP CPU variants.
//
#define FIXUP_VV_BODY            \
  Index_type ith           = i ; \
  Real_ptr vv              = vv_pair + nvals_per_std_intsc * ith ; \
  Real_const_ptr vv_int_p  = vv_int  + 72*ith ; \
  Index_type constexpr nvp = nvals_per_pair ; \
  Index_type constexpr nvb = n_vvint_per_block ; \
  Int_type k=0 ; \
  if ( 8*ith + k < n_szpairs ) { \
    vv[nvp*k+0] = vv_int_p[nvb*k+0] + vv_int_p[nvb*(k+1)+0] ;   \
    vv[nvp*k+1] = vv_int_p[nvb*k+1] + vv_int_p[nvb*(k+1)+1] ;   \
    vv[nvp*k+2] = vv_int_p[nvb*k+2] + vv_int_p[nvb*(k+1)+2] ;   \
    vv[nvp*k+3] = vv_int_p[nvb*k+3] + vv_int_p[nvb*(k+1)+3] ;   \
  } \
  for ( Index_type k=1 ; k<8 ; ++k ) { \
    if ( 8*ith + k < n_szpairs ) { \
      vv[nvp*k+0] = vv_int_p[nvb*k+4] + vv_int_p[nvb*(k+1)+0] ;  \
      vv[nvp*k+1] = vv_int_p[nvb*k+5] + vv_int_p[nvb*(k+1)+1] ;  \
      vv[nvp*k+2] = vv_int_p[nvb*k+6] + vv_int_p[nvb*(k+1)+2] ;  \
      vv[nvp*k+3] = vv_int_p[nvb*k+7] + vv_int_p[nvb*(k+1)+3] ;  \
    } \
  }



#endif // close include guard RAJAPerf_Apps_INTSC_HEXHEX_BODY_HPP
