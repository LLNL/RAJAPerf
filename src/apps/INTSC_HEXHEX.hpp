//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Intersection between two 24-sided hexahedrons, volume and moments.
///
///  for ( itri=0 ; itri < 576*n_std_intsc ; ++itri ) {
///   long const n_dsz_tris = 12 ;
///   long const n_tsz_tets = 6 ;
///   long const tri_per_pair = n_dsz_tris * n_tsz_tets ;
///   long ipair   = itri / tri_per_pair ;
///   int dfacet  = ( itri / n_tsz_tets ) % n_dsz_tris ;
///   int ttet    = itri % n_tsz_tets ;
///   long pair_base_thr = ipair * tri_per_pair ;
///   long blk_base = blk * blksize ;
///   double vv=0.0, vx=0.0, vy=0.0, vz=0.0 ;
///
///   double const *xds = dsubz + 24*ipair ;
///   double const *xts = tsubz + 24*ipair ;
///   hex_intsc_subz ( xds, xts, dfacet, ttet, vv, vx, vy, vz ) ;
///
///   vv_out[4*ipair+0]  = vv ;
///   vv_out[4*ipair+1]  = vx ;
///   vv_out[4*ipair+2]  = vy ;
///   vv_out[4*ipair+3]  = vz ;
///  }
///
/// void hex_intsc_subz
///     ( double const *xds, double const *xts,
///       int const dfacet, int const ttet,
///       double &vv, double &vx, double &vy, double &vz )
///    {
///      double const *yds = xds + 8 ;
///      double const *zds = yds + 8 ;
///      double const *yts = xts + 8 ;
///      double const *zts = yts + 8 ;
///
///      double xdt[3], ydt[3], zdt[3] ;
///      copy_donor_triangle ( xdt, ydt, zdt, xds, yds, zds ) ;
///      double xtt[4], ytt[4], ztt[4] ;
///      copy_target_tet ( xtt, ytt, ztt, xts, yts, zts ) ;
///
///      cuda_intsc_tri_tet
///       ( xdt, ydt, zdt, xtt, ytt, ztt, vv, vx, vy, vz ) ;
///    }


#ifndef RAJAPerf_Apps_INTSC_HEXHEX_HPP
#define RAJAPerf_Apps_INTSC_HEXHEX_HPP


#include "RAJA/RAJA.hpp"

#include "common/RPTypes.hpp"

namespace rajaperf {

// GPU block size is 64 for this kernel.
static Index_type constexpr gpu_block_size = 64 ;

// Number of donor triangles in a 12 sided hexahedron subzone
static Index_type constexpr n_dsz_tris = 12 ;

// Number of target tets in a 12 sided hexahedron subzone
static Index_type constexpr n_tsz_tets = 6 ;

// Number of triangle-tet contributions per subzone pair (= 72)
static Index_type constexpr tri_per_pair = n_dsz_tris * n_tsz_tets ;

// A standard intersection is defined as eight subzone pairs.
static Index_type constexpr npairs_per_std_intsc = 8 ;

// Number of subzone pairs grouped together in fixup.
//  (is gpu_block_size / gcd(gpu_block_size, tri_per_pair)
static Index_type constexpr fixup_groupsize = 8 ;

// 576 triangle contributions per group (eight subzone intersections grouped)
static Index_type constexpr tri_per_group =
    tri_per_pair * fixup_groupsize ;

// 576 threads (triangle contributions) per standard intersection.
static Index_type constexpr tri_per_std_intsc =
    tri_per_pair * npairs_per_std_intsc ;

// Compute four values per pair (intersection volume, x, y, z moments).
static Index_type constexpr nvals_per_pair = 4 ;

// Number of values computed per standard intersection ( = 4*8 = 32)
static Index_type constexpr nvals_per_std_intsc =
    nvals_per_pair * npairs_per_std_intsc ;

// 9 gpu blocks per group of intersections of subzones
static Index_type blks_per_group = tri_per_group / gpu_block_size ;

// Minimum warp size (is 32 for Cuda and 64 for Hip).
static Index_type constexpr min_warp_size = 32 ;

// Maximum warps per block (= 2)
static Index_type constexpr max_warps_per_block =
    (gpu_block_size + min_warp_size - 1) / min_warp_size ;

// Maximum number of pairs represented in a gpu block.
// With gpu_block_size < tri_per_pair at most two pairs represented in a block.
//
static Index_type constexpr max_pairs_per_block = 2 ;

// Number of intermediate computed values per gpu block
static Index_type constexpr n_vvint_per_block =
    max_pairs_per_block * nvals_per_pair ;

// 72 data entries in group in intermediate results (before fixup).
static Index_type n_vvint_per_group = n_vvint_per_block * blks_per_group ;

// shared memory size for reduction within a block.
// The two distinct subzone pairs in a block are distinct.
// For each subzone pair, have reduction values per warp.
// len_vv_reduce = 16.
static Index_type constexpr len_vv_reduce =
    max_warps_per_block * max_pairs_per_block * nvals_per_pair ;

}  // end namespace rajaperf

#define  INTSC_HEXHEX_DATA_SETUP  \
  Real_ptr const dsubz  = m_dsubz ;  \
  Real_ptr const tsubz  = m_tsubz ;  \
  Real_ptr       vv_int = m_vv_int ; \
  Real_ptr      vv_pair = m_vv_out ;

#include "INTSC_HEXHEX_BODY.hpp"

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{

class INTSC_HEXHEX : public KernelBase
{
public:

  INTSC_HEXHEX(const RunParams& params);

  ~INTSC_HEXHEX();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  void check_intsc_volume_moments(Real_const_ptr vv, VariantID vid);

  Real_type shiftTarget ( Int_type which_coord, Int_type iseq ) ;
  void exactVolMoments
      ( Int_type const kx, Int_type const ky, Int_type const kz,
        Real_type &v0, Real_type &vx, Real_type &vy, Real_type &vz ) ;

  static const size_t default_gpu_block_size = 64;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Size_type m_subz_side_length ;  // cube root of m_n_subz_intsc, even integer

  Size_type m_n_std_intsc; // number of standard intersections
  Index_type m_n_subz_intsc; // number of subzone intersections
  Index_type m_gsize;

  Real_ptr m_dsubz ;    // donor subzone coordinates
  Real_ptr m_tsubz ;    // target subzone coordinates

  Index_type m_nthreads ;     //  number of threads (=576 per standard intsc.)
  Real_ptr m_vv_int ;   // intermediate volumes and moments
  Real_ptr m_vv_out ;   // [4*n_nitsc] computed volumes, moments on device

  Real_ptr m_vv ;       // [4*n_intsc] computed volumes, moments on host

  static constexpr Real_type m_xmin = -0.2 ;  // coordinates of hex corners
  static constexpr Real_type m_xmax = -0.1 ;
  static constexpr Real_type m_ymin =  0.1 ;
  static constexpr Real_type m_ymax =  0.2 ;
  static constexpr Real_type m_zmin = -0.8 ;
  static constexpr Real_type m_zmax = -0.7 ;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
