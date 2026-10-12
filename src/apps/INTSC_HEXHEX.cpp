//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXHEX.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>
#include <iomanip>


namespace rajaperf
{
namespace apps
{


INTSC_HEXHEX::INTSC_HEXHEX(const RunParams& params)
  : KernelBase(rajaperf::Apps_INTSC_HEXHEX, params)
{
  //  one standard intersection = eight subzone intersections.
  //  Set number of standard intersections here.
  //
  //  Default number of standard intersections = 25 cubed, so as to
  //  finish the "sequential" test in one second.  The gpu tests will
  //  take only a few milliseconds for the same problem.
  //
  constexpr Size_type a3_def = 25 ;
  constexpr Size_type n_std_intsc_def = a3_def*a3_def*a3_def ;
  setDefaultProblemSize(n_std_intsc_def);

  setDefaultReps  (1);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(1);
  setProblemDimensionality(3);

  setUsesFeature(Forall);

  addVariantTunings();
}

void INTSC_HEXHEX::setSize(Index_type target_size, Index_type target_reps)
{
  // Number of standard intersections, by convention a cube number.
  Size_type side_length =
      (Size_type) ( std::cbrt((Real_type) target_size + 0.5) );

  if ( side_length < 1UL ) { side_length = 1UL ; }

  m_n_std_intsc = side_length * side_length * side_length ;

  // One standard intersection is 8 subzone intersections.
  m_n_subz_intsc = npairs_per_std_intsc * m_n_std_intsc ;

  // 8 subzones is two subzones on each side.
  m_subz_side_length = 2 * side_length ;

  const Int_type block_size = default_gpu_block_size ;
  m_nthreads = tri_per_pair * m_n_subz_intsc ;  // 72 threads per subzone pair
  m_gsize    = RAJA_DIVIDE_CEILING_INT(m_nthreads, block_size) ;

  setActualProblemSize( m_n_std_intsc ) ;
  setRunReps( target_reps );

  setItsPerRep( m_n_std_intsc );
  setKernelsPerRep(2);   // main intersection kernel and final fixup.

  setBytesAllocatedPerRep( 1*sizeof(Real_type) * 24L*m_n_subz_intsc + // dsubz
                           1*sizeof(Real_type) * 24L*m_n_subz_intsc + // tsubz
                           1*sizeof(Real_type) * n_vvint_per_block * m_gsize + // vv_int
                           1*sizeof(Real_type) * nvals_per_pair * m_n_subz_intsc + // vv_out
                           1*sizeof(Real_type) * nvals_per_pair * m_n_subz_intsc ); // vv

  // touched data size, not actual number of stores and loads
  // see VOL3D.cpp

  //  Donor and target each 24 doubles (subzone coordinates)
  //  Fixup kernel reads 72 doubles per standard intersection,
  //    or 9 doubles per subzone intersection.
  setBytesReadPerRep( (24+24+9)*8*sizeof(Real_type) * getItsPerRep() );

  // Bytes written = 9 doubles per subzone intersection includes
  //   vv_lo and vv_hi (intermediate) + 4 doubles per subzone
  //   intersection (final) = 13 doubles for a subzone intersection.
  //   A standard intersection is 8 subzone intersections.
  //
  setBytesWrittenPerRep( 13*8*sizeof(Real_type) * getItsPerRep() );
  setBytesModifyWrittenPerRep( 0 );
  setBytesAtomicModifyWrittenPerRep( 0 );

  constexpr Size_type flops_per_tri = 336 ;
  constexpr Size_type flops_per_intsc = flops_per_tri * tri_per_std_intsc ;

  setFLOPsPerRep(m_n_std_intsc * flops_per_intsc);
}

INTSC_HEXHEX::~INTSC_HEXHEX()
{
}


Real_type INTSC_HEXHEX::shiftTarget
    ( Int_type which_coord,   //  0=X, 1=Y, 2=Z
      Int_type iseq )     // sequence number of intersection along the "side"
{
  // phase multiplier for sin, ~2Pi * (2-golden ratio)
  Real_type constexpr q = 2.3999632297286533 ;
  Real_type  width ;

  switch ( which_coord ) {
  case 0:    width = m_xmax - m_xmin ; break ;
  case 1:    width = m_ymax - m_ymin ; break ;
  case 2:    width = m_zmax - m_zmin ; break ;
  default:
    width = 0.0 ;
    getCout() << "INTSC_HEXHEX::shift_target : Error in which_coord\n" <<
      std::endl ;
  }

  // The 0.95 ensures at least 5% overlap each direction.
  Real_type shift = 0.95 * sin ( q * ((Real_type)iseq + 0.5) ) * width ;

  return shift ;
}


void INTSC_HEXHEX::setUp(VariantID vid,
                         Size_type RAJAPERF_UNUSED_ARG(tune_idx))
{
  // coordinates for donor subzone (the eight corner points)
  Real_type xdzone[8] =
      { m_xmin, m_xmax, m_xmin, m_xmax, m_xmin, m_xmax, m_xmin, m_xmax } ;

  Real_type ydzone[8] =
      { m_ymin, m_ymin, m_ymax, m_ymax, m_ymin, m_ymin, m_ymax, m_ymax } ;

  Real_type zdzone[8] =
      { m_zmin, m_zmin, m_zmin, m_zmin, m_zmax, m_zmax, m_zmax, m_zmax } ;

  auto a_ds = allocDataForInit ( m_dsubz, 24L*m_n_subz_intsc, vid ) ;
  auto a_ts = allocDataForInit ( m_tsubz, 24L*m_n_subz_intsc, vid ) ;

  //  Set up donor and target coordinates for each subzone intersection.
  //  Same donor coordinates for all intersections,
  //  target coordinates vary.
  Index_type k = 0 ;
  for ( Size_type kz=0 ; kz < m_subz_side_length ; ++kz ) {
    for ( Size_type ky=0 ; ky < m_subz_side_length ; ++ky ) {
      for ( Size_type kx=0 ; kx < m_subz_side_length ; ++kx ) {
        for ( Index_type i=0 ; i<8 ; ++i ) {
          m_dsubz[24L*k+ 0+i] = xdzone[i] ;
          m_dsubz[24L*k+ 8+i] = ydzone[i] ;
          m_dsubz[24L*k+16+i] = zdzone[i] ;
          m_tsubz[24L*k+ 0+i] = xdzone[i] + shiftTarget(0, kx) ;
          m_tsubz[24L*k+ 8+i] = ydzone[i] + shiftTarget(1, ky) ;
          m_tsubz[24L*k+16+i] = zdzone[i] + shiftTarget(2, kz) ;
        }
        ++k ;
      }
    }
  }

  // intermediate volumes, moments
  allocData ( m_vv_int, n_vvint_per_block * m_gsize, vid ) ;

  allocAndInitDataConst ( m_vv_out, nvals_per_pair * m_n_subz_intsc, 0.0, vid ) ;

  // output volumes and moments on the host
  allocData ( DataSpace::Host, m_vv, nvals_per_pair * m_n_subz_intsc ) ;
}


//  Get exact volume and moments of Cartesian aligned zone intersections.
void INTSC_HEXHEX::exactVolMoments
    ( Int_type const kx,   //  x index of the subzone
      Int_type const ky,   //  y index of the subzone
      Int_type const kz,   //  z index of the subzone
      Real_type &v0,       // exact intersection volume
      Real_type &vx,       // exact intersection x moment
      Real_type &vy,       // exact intersection y moment
      Real_type &vz )      // exact intersection z moment
{
  Real_type xshift = shiftTarget (0, kx) ;
  Real_type yshift = shiftTarget (1, ky) ;
  Real_type zshift = shiftTarget (2, kz) ;

  Real_type xmin = ( xshift > 0.0 ) ? m_xmin + xshift : m_xmin ;
  Real_type ymin = ( yshift > 0.0 ) ? m_ymin + yshift : m_ymin ;
  Real_type zmin = ( zshift > 0.0 ) ? m_zmin + zshift : m_zmin ;

  Real_type xmax = ( xshift > 0.0 ) ? m_xmax : m_xmax + xshift ;
  Real_type ymax = ( yshift > 0.0 ) ? m_ymax : m_ymax + yshift ;
  Real_type zmax = ( zshift > 0.0 ) ? m_zmax : m_zmax + zshift ;

  Real_type dx = xmax - xmin ;
  Real_type dy = ymax - ymin ;
  Real_type dz = zmax - zmin ;
  if ( dx <= 0.0 || dy <= 0.0 || dz <= 0.0 ) {
    v0 = vx = vy = vz = 0.0 ;
  } else {
    Real_type xc = 0.5 * ( xmax + xmin ) ;
    Real_type yc = 0.5 * ( ymax + ymin ) ;
    Real_type zc = 0.5 * ( zmax + zmin ) ;

    v0 = dx * dy * dz ;
    vx = v0 * xc ;
    vy = v0 * yc ;
    vz = v0 * zc ;
  }
}



//   Number of subzone intersections = 8 * number of standard intersections.
//
void INTSC_HEXHEX::check_intsc_volume_moments(
    Real_const_ptr vv,   // computed volumes, moments on the host
    VariantID vid)       // Print variant name in case of error
{

  {
    Char_const_ptr tst = "INTSC_HEXHEX:" ;

    // volume of the donor and target subzones (all the same)
    Real_type zvol =
        ( m_xmax - m_xmin ) *
        ( m_ymax - m_ymin ) *
        ( m_zmax - m_zmin ) ;

    Real_type tolsq = 1.0e-24 ;
    Real_type tolsqv = tolsq * zvol*zvol ;
    Real_type tolsqx = tolsq * zvol*zvol *
        ( fabs(m_xmax) + fabs(m_xmin) ) * ( fabs(m_xmax) + fabs(m_xmin) ) ;
    Real_type tolsqy = tolsq * zvol*zvol *
        ( fabs(m_ymax) + fabs(m_ymin) ) * ( fabs(m_ymax) + fabs(m_ymin) ) ;
    Real_type tolsqz = tolsq * zvol*zvol *
        ( fabs(m_zmax) + fabs(m_zmin) ) * ( fabs(m_zmax) + fabs(m_zmin) ) ;

    Index_type k=0 ;

    for ( Size_type kz=0 ; kz < m_subz_side_length ; ++kz ) {
      for ( Size_type ky=0 ; ky < m_subz_side_length ; ++ky ) {
        for ( Size_type kx=0 ; kx < m_subz_side_length ; ++kx ) {

          //   The correct volume and moments.
          Real_type v0, vx, vy, vz ;

          exactVolMoments ( kx, ky, kz, v0, vx, vy, vz ) ;

          //  differences between computed and correct
          Real_type dv  = vv[ nvals_per_pair*k + 0 ] - v0 ;
          Real_type dxm = vv[ nvals_per_pair*k + 1 ] - vx ;
          Real_type dym = vv[ nvals_per_pair*k + 2 ] - vy ;
          Real_type dzm = vv[ nvals_per_pair*k + 3 ] - vz ;

          // Print an error message if a volume or moment is incorrect.
          if ( ( dv*dv   > tolsqv ) ||
               ( dxm*dxm > tolsqx ) ||
               ( dym*dym > tolsqy ) ||
               ( dzm*dzm > tolsqz ) ) {

            auto show_comparison = [&]
                ( Int_type  kintsc,
                  std::string lbl,
                  Real_type vcalc,
                  Real_type const vexpected,
                  Real_type const tol )
                                   {
                                     getCout()
                                         << tst << " k = " << kintsc
                                         << "    " << lbl << " = "
                                         << std::scientific
                                         << std::setprecision(15)
                                         << std::setw(23) << vcalc
                                         << "  expected "
                                         << std::setw(23) << vexpected
                                         << "   tolerance"
                                         << std::setprecision(3)
                                         << std::setw(12) << tol
                                         << std::endl ;
                                   } ;

            getCout()
                << tst
                << " Calculated Volumes and/or moments are INCORRECT for "
                << getVariantName(vid).c_str() << "." << std::endl
                << tst
                << " First error encountered:" << std::endl ;

            show_comparison
                ( k, "vv", vv[nvals_per_pair*k+0], v0, sqrt(tolsqv) ) ;
            show_comparison
                ( k, "vx", vv[nvals_per_pair*k+1], vx, sqrt(tolsqx) ) ;
            show_comparison
                ( k, "vy", vv[nvals_per_pair*k+2], vy, sqrt(tolsqy) ) ;
            show_comparison
                ( k, "vz", vv[nvals_per_pair*k+3], vz, sqrt(tolsqz) ) ;
            getCout() << std::endl ;

            break ;
          }
          ++k ;
        }
      }
    }
  }
}


void INTSC_HEXHEX::updateChecksum(VariantID vid,
                                  size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  copyData ( DataSpace::Host, m_vv,
             getDataSpace(vid), m_vv_out, nvals_per_pair*m_n_subz_intsc ) ;

  check_intsc_volume_moments ( m_vv, vid ) ;

  addToChecksum(m_vv_out, nvals_per_pair*m_n_subz_intsc, vid);
}

void INTSC_HEXHEX::tearDown(VariantID vid,
                            Size_type RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData ( m_dsubz, vid ) ;
  deallocData ( m_tsubz, vid ) ;
  deallocData ( m_vv_int, vid ) ;
  deallocData ( m_vv_out, vid ) ;
  deallocData ( DataSpace::Host, m_vv ) ;
}

} // end namespace apps
} // end namespace rajaperf
