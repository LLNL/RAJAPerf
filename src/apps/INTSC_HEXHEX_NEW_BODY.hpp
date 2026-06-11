//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Apps_INTSC_HEXHEX_NEW_BODY_HPP
#define RAJAPerf_Apps_INTSC_HEXHEX_NEW_BODY_HPP

namespace rajaperf {

constexpr int hexhex_new_max_poly_vertices = 9;

// Initial polygon vertex order
static constexpr unsigned long long HEXHEX_NEW_NEXT_INIT = 0xF87654F21ULL;

enum class HexHexPackedPlaneNew {
  H2,   // 1 - x - y >= 0
  X,    // x >= 0
  Y,    // y >= 0
  Z,    // z >= 0
  H,    // h >= 0
  NEG_H // -h - eps >= 0
};

RAJA_HOST_DEVICE
RAJA_INLINE Int_type hexhex_new_next_get(unsigned long long pack,
                                         Int_type idx) {
  unsigned int const v =
      static_cast<unsigned int>((pack >> (4 * idx)) & 0xFULL);

  return (v == 0xFULL) ? Int_type(-1) : Int_type(v);
}

RAJA_HOST_DEVICE
RAJA_INLINE void hexhex_new_next_set(unsigned long long &pack, Int_type idx,
                                     Int_type next) {
  unsigned long long const shift = static_cast<unsigned long long>(4 * idx);
  unsigned long long const mask = 0xFULL << shift;

  unsigned long long const v = static_cast<unsigned long long>(
      next < 0 ? 0xFULL : static_cast<unsigned int>(next));

  pack = (pack & ~mask) | ((v & 0xFULL) << shift);
}

// Plane representation to reduce local memory
template <HexHexPackedPlaneNew P>
RAJA_HOST_DEVICE RAJA_INLINE Real_type hexhex_plane_value_packed_new(
    Real_type const x[hexhex_new_max_poly_vertices],
    Real_type const y[hexhex_new_max_poly_vertices],
    Real_type const z[hexhex_new_max_poly_vertices], Int_type const j) {
  switch (P) {
  case HexHexPackedPlaneNew::H2:
    return Real_type(1.0) - x[j] - y[j];
  case HexHexPackedPlaneNew::X:
    return x[j];
  case HexHexPackedPlaneNew::Y:
    return y[j];
  case HexHexPackedPlaneNew::Z:
    return z[j];
  case HexHexPackedPlaneNew::H:
    return Real_type(1.0) - x[j] - y[j] - z[j];
  case HexHexPackedPlaneNew::NEG_H:
    return x[j] + y[j] + z[j] - Real_type(1.0) - Real_type(1.0e-50);
  }

  return Real_type(0.0);
}

// only interpolate x,y,z to avoid saving h
RAJA_HOST_DEVICE
RAJA_INLINE void
hexhex_interp_zero_packed_new(Real_type x[hexhex_new_max_poly_vertices],
                              Real_type y[hexhex_new_max_poly_vertices],
                              Real_type z[hexhex_new_max_poly_vertices],
                              Int_type out, Int_type a, Int_type b,
                              Real_type ca, Real_type cb) {
  Real_type t = ca / (ca - cb);
  Real_type omt = Real_type(1.0) - t;

  x[out] = x[a] * omt + x[b] * t;
  y[out] = y[a] * omt + y[b] * t;
  z[out] = z[a] * omt + z[b] * t;
}

template <HexHexPackedPlaneNew P>
RAJA_HOST_DEVICE RAJA_INLINE void
clip_polygon_ge_0_packed_new(Real_type x[hexhex_new_max_poly_vertices],
                             Real_type y[hexhex_new_max_poly_vertices],
                             Real_type z[hexhex_new_max_poly_vertices],
                             Int_type &first, Int_type &avail,
                             unsigned long long &next_pack) {
  Int_type j = first;

  if (j < 0) {
    return;
  }

  Int_type first0 = first;
  Int_type j1 = -1, j2 = -1;
  Int_type jj1 = -1, jj2 = -1;

  Real_type c0 = hexhex_plane_value_packed_new<P>(x, y, z, j);
  Real_type c00 = c0;
  Real_type clast = c0;

// gives better performance that aggressive loop unrolling
#pragma unroll 1
  while (j >= 0) {
    Int_type jj = hexhex_new_next_get(next_pack, j);
    Int_type jp = jj;
    if (jj < 0) {
      jj = first0;
    }

    Real_type c1 = hexhex_plane_value_packed_new<P>(x, y, z, jj);
    if ((c0 >= Real_type(0.0)) && (c1 < Real_type(0.0))) {
      j1 = j;
      jj1 = jj;
    }
    if ((c0 < Real_type(0.0)) && (c1 >= Real_type(0.0))) {
      j2 = j;
      jj2 = jj;
    }
    j = jp;
    clast = c0;
    c0 = c1;
  }

  if (j1 < 0) {
    if (c00 >= Real_type(0.0)) {
      return;
    } else {
      first = -1;
      return;
    }
  }

  Int_type jr1 = -1, jr2 = -1;

  if (j1 >= 0) {
    jr1 = avail;
    avail = hexhex_new_next_get(next_pack, avail);
    hexhex_interp_zero_packed_new(
        x, y, z, jr1, j1, jj1, hexhex_plane_value_packed_new<P>(x, y, z, j1),
        hexhex_plane_value_packed_new<P>(x, y, z, jj1));

    jr2 = avail;
    avail = hexhex_new_next_get(next_pack, avail);
    hexhex_interp_zero_packed_new(
        x, y, z, jr2, j2, jj2, hexhex_plane_value_packed_new<P>(x, y, z, j2),
        hexhex_plane_value_packed_new<P>(x, y, z, jj2));
  }

  first = -1;

  j = first0;

// gives better performance that aggressive loop unrolling
#pragma unroll 1
  while (j >= 0) {
    Int_type jp = hexhex_new_next_get(next_pack, j);
    if (hexhex_plane_value_packed_new<P>(x, y, z, j) < Real_type(0.0)) {
      hexhex_new_next_set(next_pack, j, avail);
      avail = j;
    } else if (first == -1) {
      first = j;
    }
    j = jp;
  }

  if (j1 >= 0) {
    hexhex_new_next_set(next_pack, j1, jr1);
    hexhex_new_next_set(next_pack, jr1, jr2);
    hexhex_new_next_set(next_pack, jr2,
                        ((clast < Real_type(0.0)) || (c00 < Real_type(0.0)))
                            ? Int_type(-1)
                            : jj2);
  }
}

//   Simplified volume calculation, for the area under one
//   polygonal face, on the Cuda device.
//   Planar polygon.
//   Compute volume, moments between polygon and the z=0 plane.
RAJA_HOST_DEVICE
RAJA_INLINE void cuda_hex_volpolyh_1poly_packed_new(
    Real_type const x[hexhex_new_max_poly_vertices],
    Real_type const y[hexhex_new_max_poly_vertices],
    Real_type const z[hexhex_new_max_poly_vertices], Int_type const first,
    unsigned long long const next_pack, Real_type &vv, Real_type &vx,
    Real_type &vy, Real_type &vz) {
  if (first < 0) {
    return;
  }

  Int_type const j0 = first;
  Real_type x0 = x[j0];
  Real_type y0 = y[j0];
  Real_type z0 = z[j0];

  Int_type const j1 = hexhex_new_next_get(next_pack, j0);
  if (j1 < 0) {
    return;
  }

  Real_type x1 = x[j1];
  Real_type y1 = y[j1];
  Real_type z1 = z[j1];
  Real_type dx1 = x1 - x0;
  Real_type dy1 = y1 - y0;

  Int_type j2 = hexhex_new_next_get(next_pack, j1);

#pragma unroll 1
  while (j2 >= 0) {
    Real_type x2 = x[j2];
    Real_type y2 = y[j2];
    Real_type z2 = z[j2];
    Real_type dx2 = x2 - x0;
    Real_type dy2 = y2 - y0;

    Real_type area2 = (dx1 * dy2 - dx2 * dy1);
    Real_type v0 = (z0 + z1 + z2) * area2;
    vv += v0;
    vx += v0 * (x0 + x1 + x2) + area2 * (x0 * z0 + x1 * z1 + x2 * z2);
    vy += v0 * (y0 + y1 + y2) + area2 * (y0 * z0 + y1 * z1 + y2 * z2);
    vz += (z0 * z0 + z1 * z1 + z2 * z2 + z0 * z1 + z0 * z2 + z1 * z2) * area2;

    x1 = x2;
    y1 = y2;
    z1 = z2; // Rotate.
    dx1 = dx2;
    dy1 = dy2;

    j2 = hexhex_new_next_get(next_pack, j2);
  }
}

RAJA_HOST_DEVICE
RAJA_INLINE void cuda_intsc_tri_tet_new(
    Real_type const (&xdt)[3], // donor triangle coordinates
    Real_type const (&ydt)[3], Real_type const (&zdt)[3],
    Real_type (&xtt)[4], // target tet coordinates (modified here)
    Real_type (&ytt)[4], Real_type (&ztt)[4],
    Real_type &vv_thr, // volume contribution for this triangle-tet
    Real_type &vx_thr, // x moment contribution for this triangle-tet
    Real_type &vy_thr, // y moment contribution for this triangle-tet
    Real_type &vz_thr) // z moment contribution for this triangle-tet
{
  Real_type det, deti;
  Real_type xa0, xa1, xa2;
  Real_type ya0, ya1, ya2;
  Real_type za0, za1, za2;

  Real_type vv = 0.0, vx = 0.0, vy = 0.0, vz = 0.0; // volume, moments.

  xtt[1] -= xtt[0];
  xtt[2] -= xtt[0];
  xtt[3] -= xtt[0];
  ytt[1] -= ytt[0];
  ytt[2] -= ytt[0];
  ytt[3] -= ytt[0];
  ztt[1] -= ztt[0];
  ztt[2] -= ztt[0];
  ztt[3] -= ztt[0];

  det = xtt[1] * ytt[2] * ztt[3] - xtt[1] * ytt[3] * ztt[2] +
        xtt[2] * ytt[3] * ztt[1] - xtt[2] * ytt[1] * ztt[3] +
        xtt[3] * ytt[1] * ztt[2] - xtt[3] * ytt[2] * ztt[1];
  deti = det / (det * det + 1.0e-100);

  // Cross products.
  Real_type cyz = ytt[2] * ztt[3] - ztt[2] * ytt[3];
  Real_type czx = ztt[2] * xtt[3] - xtt[2] * ztt[3];
  Real_type cxy = xtt[2] * ytt[3] - ytt[2] * xtt[3];

  //   Coordinates of the facet in the transformed frame.
  xa0 = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
        (zdt[0] - ztt[0]) * cxy;
  xa1 = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
        (zdt[1] - ztt[0]) * cxy;
  xa2 = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
        (zdt[2] - ztt[0]) * cxy;

  cyz = ytt[3] * ztt[1] - ztt[3] * ytt[1];
  czx = ztt[3] * xtt[1] - xtt[3] * ztt[1];
  cxy = xtt[3] * ytt[1] - ytt[3] * xtt[1];

  ya0 = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
        (zdt[0] - ztt[0]) * cxy;
  ya1 = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
        (zdt[1] - ztt[0]) * cxy;
  ya2 = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
        (zdt[2] - ztt[0]) * cxy;

  cyz = ytt[1] * ztt[2] - ztt[1] * ytt[2];
  czx = ztt[1] * xtt[2] - xtt[1] * ztt[2];
  cxy = xtt[1] * ytt[2] - ytt[1] * xtt[2];

  za0 = (xdt[0] - xtt[0]) * cyz + (ydt[0] - ytt[0]) * czx +
        (zdt[0] - ztt[0]) * cxy;
  za1 = (xdt[1] - xtt[0]) * cyz + (ydt[1] - ytt[0]) * czx +
        (zdt[1] - ztt[0]) * cxy;
  za2 = (xdt[2] - xtt[0]) * cyz + (ydt[2] - ytt[0]) * czx +
        (zdt[2] - ztt[0]) * cxy;

  xa0 *= deti;
  xa1 *= deti;
  xa2 *= deti;
  ya0 *= deti;
  ya1 *= deti;
  ya2 *= deti;
  za0 *= deti;
  za1 *= deti;
  za2 *= deti;

  Real_type xa[hexhex_new_max_poly_vertices];
  Real_type ya[hexhex_new_max_poly_vertices];
  Real_type za[hexhex_new_max_poly_vertices];

  xa[0] = xa0;
  xa[1] = xa1;
  xa[2] = xa2;

  ya[0] = ya0;
  ya[1] = ya1;
  ya[2] = ya2;

  za[0] = za0;
  za[1] = za1;
  za[2] = za2;

  // NOTE: early exit to see if triangle and plane are clipping

  Real_type const h20 = Real_type(1.0) - xa[0] - ya[0];
  Real_type const h21 = Real_type(1.0) - xa[1] - ya[1];
  Real_type const h22 = Real_type(1.0) - xa[2] - ya[2];

  if ((h20 < Real_type(0.0) && h21 < Real_type(0.0) && h22 < Real_type(0.0)) ||
      (xa[0] < Real_type(0.0) && xa[1] < Real_type(0.0) &&
       xa[2] < Real_type(0.0)) ||
      (ya[0] < Real_type(0.0) && ya[1] < Real_type(0.0) &&
       ya[2] < Real_type(0.0)) ||
      (za[0] < Real_type(0.0) && za[1] < Real_type(0.0) &&
       za[2] < Real_type(0.0))) {
    return;
  }

  unsigned long long next_pack = HEXHEX_NEW_NEXT_INIT;
  Int_type first = 0;
  Int_type avail = 3;

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::H2>(xa, ya, za, first,
                                                         avail, next_pack);
  if (first < 0) {
    return;
  }

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::X>(xa, ya, za, first,
                                                        avail, next_pack);
  if (first < 0) {
    return;
  }

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::Y>(xa, ya, za, first,
                                                        avail, next_pack);
  if (first < 0) {
    return;
  }

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::Z>(xa, ya, za, first,
                                                        avail, next_pack);
  if (first < 0) {
    return;
  }

  Int_type const first_saved = first;
  Int_type const avail_saved = avail;
  unsigned long long const next_saved = next_pack;

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::H>(xa, ya, za, first,
                                                        avail, next_pack);
  cuda_hex_volpolyh_1poly_packed_new(xa, ya, za, first, next_pack, vv, vx, vy,
                                     vz);

  first = first_saved;
  avail = avail_saved;
  next_pack = next_saved;

  clip_polygon_ge_0_packed_new<HexHexPackedPlaneNew::NEG_H>(xa, ya, za, first,
                                                            avail, next_pack);

#pragma unroll 1
  for (Int_type j = first; j >= 0; j = hexhex_new_next_get(next_pack, j)) {
    za[j] = Real_type(1.0) - xa[j] - ya[j];
  }

  cuda_hex_volpolyh_1poly_packed_new(xa, ya, za, first, next_pack, vv, vx, vy,
                                     vz);

  //  Volume, moments of the intersection in the unit tet frame.
  vv *= 0.16666666666666667;
  vx *= 0.041666666666666667;
  vy *= 0.041666666666666667;
  vz *= 0.041666666666666667;

  //   Transform moments to the physical frame.
  vx_thr += det * (xtt[0] * vv + xtt[1] * vx + xtt[2] * vy + xtt[3] * vz);
  vy_thr += det * (ytt[0] * vv + ytt[1] * vx + ytt[2] * vy + ytt[3] * vz);
  vz_thr += det * (ztt[0] * vv + ztt[1] * vx + ztt[2] * vy + ztt[3] * vz);

  //   Transform intersection volume to the physical frame.
  vv_thr += det * vv;
}

//  Compute the contribution of a donor triangle and a target tet
//         to intersection between hex subzones.
//   Each subzone is twelve triangular facets (six tets).
//
RAJA_HOST_DEVICE
RAJA_INLINE void hex_intsc_subz_new(
    Real_const_ptr xds,    //  [24] donor subzone coords
    Real_const_ptr xts,    //  [24] target subzone coords
    Int_type const dfacet, // which donor facet
    Int_type const ttet,   // which target tet
    Real_type &vv_thr,     // volume contribution for this triangle-tet
    Real_type &vx_thr,     // x moment contribution for this triangle-tet
    Real_type &vy_thr,     // y moment contribution for this triangle-tet
    Real_type &vz_thr)     // z moment contribution for this triangle-tet
{
  Real_const_ptr yds = xds + 8;
  Real_const_ptr zds = yds + 8;

  Real_const_ptr yts = xts + 8;
  Real_const_ptr zts = yts + 8;

  vv_thr = 0.0;
  vx_thr = 0.0;
  vy_thr = 0.0;
  vz_thr = 0.0;

  Int_type const n_dfacets = 12;
  Int_type const len_cycnod = n_dfacets / 2 + 1;

  //  coordinates of the donor triangle
  Real_type xdt[3], ydt[3], zdt[3];

  {
    //  cyclic nodes to form facets with node 0.
    Int_type cyc_nod[len_cycnod] = {1, 5, 4, 6, 2, 3, 1};

    // which subzone vertices form the triangular facet.
    Int_type v0, v1, v2;
    if (dfacet < 6) {
      v0 = 0;
      v1 = cyc_nod[dfacet];
      v2 = cyc_nod[dfacet + 1];
    } else {
      v0 = 7;
      v1 = cyc_nod[n_dfacets - dfacet];
      v2 = cyc_nod[n_dfacets - dfacet - 1]; // reverse order
    }

    //  Donor triangle coordinates.
    xdt[0] = xds[v0]; // Donor facet vertices
    xdt[1] = xds[v1];
    xdt[2] = xds[v2];
    ydt[0] = yds[v0];
    ydt[1] = yds[v1];
    ydt[2] = yds[v2];
    zdt[0] = zds[v0];
    zdt[1] = zds[v1];
    zdt[2] = zds[v2];
  }

  //   Set up the target tet and do the intersections.

  Real_type xtt[4], ytt[4], ztt[4];

  xtt[0] = xts[0];
  ytt[0] = yts[0];
  ztt[0] = zts[0];

  //  subzone vertices that form the cycle for tets.
  Int_type vert_cyc[6] = {1, 3, 2, 6, 4, 5};

  Int_type v1 = vert_cyc[ttet];
  xtt[1] = xts[v1];
  ytt[1] = yts[v1];
  ztt[1] = zts[v1];
  Int_type v2 = vert_cyc[(ttet + 1) % 6];
  xtt[2] = xts[v2];
  ytt[2] = yts[v2];
  ztt[2] = zts[v2];
  xtt[3] = xts[7];
  ytt[3] = yts[7];
  ztt[3] = zts[7];

  cuda_intsc_tri_tet_new(xdt, ydt, zdt, xtt, ytt, ztt, vv_thr, vx_thr, vy_thr,
                         vz_thr);
}

} // end namespace rajaperf

#define INTSC_HEXHEX_NEW_BODY_SEQ                                              \
  Index_type ipair = ith / hexhex_new_tri_per_pair;                            \
  Int_type dfacet = (ith / hexhex_new_n_tsz_tets) % hexhex_new_n_dsz_tris;     \
  Int_type ttet = ith % hexhex_new_n_tsz_tets;                                 \
  Index_type pair_base_thr = ipair * hexhex_new_tri_per_pair;                  \
  Index_type blk_base = blk * blksize;                                         \
  Real_type vv_lo = 0.0, vx_lo = 0.0, vy_lo = 0.0, vz_lo = 0.0;                \
  Real_type vv_hi = 0.0, vx_hi = 0.0, vy_hi = 0.0, vz_hi = 0.0;                \
  if (ipair < nisc_stage) {                                                    \
    Real_const_ptr xds = dsubz + 24 * ipair;                                   \
    Real_const_ptr xts = tsubz + 24 * ipair;                                   \
    hex_intsc_subz_new(xds, xts, dfacet, ttet, vv_lo, vx_lo, vy_lo, vz_lo);    \
  }                                                                            \
  if (pair_base_thr > blk_base) {                                              \
    vv_hi = vv_lo;                                                             \
    vx_hi = vx_lo;                                                             \
    vy_hi = vy_lo;                                                             \
    vz_hi = vz_lo;                                                             \
    vv_lo = 0.0;                                                               \
    vx_lo = 0.0;                                                               \
    vy_lo = 0.0;                                                               \
    vz_lo = 0.0;                                                               \
  }

// thridx is threadIdx.x

#define INTSC_HEXHEX_NEW_BODY                                                  \
  INTSC_HEXHEX_NEW_BODY_SEQ                                                    \
                                                                               \
  __syncthreads();                                                             \
  for (Index_type k = 1; k < RAJAPERF_HEXHEX_WARPSIZE; k *= 2) {               \
    vv_hi += RAJAPERF_HEXHEX_shfl_xor(vv_hi, k);                               \
    vx_hi += RAJAPERF_HEXHEX_shfl_xor(vx_hi, k);                               \
    vy_hi += RAJAPERF_HEXHEX_shfl_xor(vy_hi, k);                               \
    vz_hi += RAJAPERF_HEXHEX_shfl_xor(vz_hi, k);                               \
    vv_lo += RAJAPERF_HEXHEX_shfl_xor(vv_lo, k);                               \
    vx_lo += RAJAPERF_HEXHEX_shfl_xor(vx_lo, k);                               \
    vy_lo += RAJAPERF_HEXHEX_shfl_xor(vy_lo, k);                               \
    vz_lo += RAJAPERF_HEXHEX_shfl_xor(vz_lo, k);                               \
  }                                                                            \
  Int_type const nwarps = blksize / RAJAPERF_HEXHEX_WARPSIZE;                  \
  Int_type k = thridx / RAJAPERF_HEXHEX_WARPSIZE;                              \
  if (thridx == k * RAJAPERF_HEXHEX_WARPSIZE) {                                \
    vv_reduce[k + 0 * hexhex_new_max_warps_per_block] = vv_lo;                 \
    vv_reduce[k + 1 * hexhex_new_max_warps_per_block] = vx_lo;                 \
    vv_reduce[k + 2 * hexhex_new_max_warps_per_block] = vy_lo;                 \
    vv_reduce[k + 3 * hexhex_new_max_warps_per_block] = vz_lo;                 \
    vv_reduce[k + 4 * hexhex_new_max_warps_per_block] = vv_hi;                 \
    vv_reduce[k + 5 * hexhex_new_max_warps_per_block] = vx_hi;                 \
    vv_reduce[k + 6 * hexhex_new_max_warps_per_block] = vy_hi;                 \
    vv_reduce[k + 7 * hexhex_new_max_warps_per_block] = vz_hi;                 \
  }                                                                            \
  __syncthreads();                                                             \
  if (thridx < hexhex_new_max_pairs_per_block * hexhex_new_nvals_per_pair) {   \
    for (Index_type k = 1; k < nwarps; ++k) {                                  \
      vv_reduce[hexhex_new_max_warps_per_block * thridx] +=                    \
          vv_reduce[hexhex_new_max_warps_per_block * thridx + k];              \
    }                                                                          \
    vv_int_p[thridx] = vv_reduce[hexhex_new_max_warps_per_block * thridx];     \
  }

#define INTSC_HEXHEX_NEW_SEQ(i, iend)                                          \
  Index_type nisc_stage = iend;                                                \
  Index_type blksize = default_gpu_block_size;                                 \
  Index_type ith = i;                                                          \
  Index_type blk = ith / blksize;                                              \
  Real_ptr vv_int_p = vv_int + hexhex_new_n_vvint_per_block * blk;             \
  if (i == 0) {                                                                \
    Index_type gsize = iend / blksize;                                         \
    Index_type vv_int_len = hexhex_new_n_vvint_per_block * gsize;              \
    for (Index_type k = 0; k < vv_int_len; ++k) {                              \
      vv_int_p[k] = 0.0;                                                       \
    }                                                                          \
  }                                                                            \
  INTSC_HEXHEX_NEW_BODY_SEQ;                                                   \
  vv_int_p[0] += vv_lo;                                                        \
  vv_int_p[1] += vx_lo;                                                        \
  vv_int_p[2] += vy_lo;                                                        \
  vv_int_p[3] += vz_lo;                                                        \
  vv_int_p[4] += vv_hi;                                                        \
  vv_int_p[5] += vx_hi;                                                        \
  vv_int_p[6] += vy_hi;                                                        \
  vv_int_p[7] += vz_hi;

//  Index i is standard intersection, ipair0 = 8*i is the first
//  subzone pair for this intersection.  Initializes 32 output values
//  for the eight pairs in the first loop.
//
#define INTSC_HEXHEX_NEW_OMP(i, iend)                                          \
  Index_type blksize = default_gpu_block_size;                                 \
  Index_type nisc_stage = iend * hexhex_new_tri_per_group;                     \
  Real_ptr vv_int_p0 = vv_int + i * hexhex_new_n_vvint_per_group;              \
  for (Index_type j = 0; j < hexhex_new_n_vvint_per_group; ++j) {              \
    vv_int_p0[j] = 0.0;                                                        \
  }                                                                            \
  Index_type j0 = i * hexhex_new_tri_per_group;                                \
  for (Index_type j = 0; j < hexhex_new_tri_per_group; ++j) {                  \
    Index_type ith = j0 + j;                                                   \
    Index_type blk = ith / blksize;                                            \
    INTSC_HEXHEX_NEW_BODY_SEQ;                                                 \
    Real_ptr vv_int_p = vv_int + hexhex_new_n_vvint_per_block * blk;           \
    vv_int_p[0] += vv_lo;                                                      \
    vv_int_p[1] += vx_lo;                                                      \
    vv_int_p[2] += vy_lo;                                                      \
    vv_int_p[3] += vz_lo;                                                      \
    vv_int_p[4] += vv_hi;                                                      \
    vv_int_p[5] += vx_hi;                                                      \
    vv_int_p[6] += vy_hi;                                                      \
    vv_int_p[7] += vz_hi;                                                      \
  }

//  This is not needed on Seq and OMP CPU variants.
//
#define INTSC_HEXHEX_NEW_FIXUP_VV_BODY                                         \
  Index_type ith = i;                                                          \
  Real_ptr vv = vv_pair + hexhex_new_nvals_per_std_intsc * ith;                \
  Real_const_ptr vv_int_p = vv_int + 72 * ith;                                 \
  Index_type constexpr nvp = hexhex_new_nvals_per_pair;                        \
  Index_type constexpr nvb = hexhex_new_n_vvint_per_block;                     \
  Int_type k = 0;                                                              \
  if (8 * ith + k < n_szpairs) {                                               \
    vv[nvp * k + 0] = vv_int_p[nvb * k + 0] + vv_int_p[nvb * (k + 1) + 0];     \
    vv[nvp * k + 1] = vv_int_p[nvb * k + 1] + vv_int_p[nvb * (k + 1) + 1];     \
    vv[nvp * k + 2] = vv_int_p[nvb * k + 2] + vv_int_p[nvb * (k + 1) + 2];     \
    vv[nvp * k + 3] = vv_int_p[nvb * k + 3] + vv_int_p[nvb * (k + 1) + 3];     \
  }                                                                            \
  for (Index_type k = 1; k < 8; ++k) {                                         \
    if (8 * ith + k < n_szpairs) {                                             \
      vv[nvp * k + 0] = vv_int_p[nvb * k + 4] + vv_int_p[nvb * (k + 1) + 0];   \
      vv[nvp * k + 1] = vv_int_p[nvb * k + 5] + vv_int_p[nvb * (k + 1) + 1];   \
      vv[nvp * k + 2] = vv_int_p[nvb * k + 6] + vv_int_p[nvb * (k + 1) + 2];   \
      vv[nvp * k + 3] = vv_int_p[nvb * k + 7] + vv_int_p[nvb * (k + 1) + 3];   \
    }                                                                          \
  }

#endif // close include guard RAJAPerf_Apps_INTSC_HEXHEX_NEW_BODY_HPP
