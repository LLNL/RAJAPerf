//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXHEX_NEW.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>
#include <iomanip>


namespace rajaperf
{
namespace apps
{


INTSC_HEXHEX_NEW::INTSC_HEXHEX_NEW(const RunParams& params)
  : KernelBase(rajaperf::Apps_INTSC_HEXHEX_NEW, params)
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

void INTSC_HEXHEX_NEW::setSize(Index_type target_size, Index_type target_reps)
{
  // Number of standard intersections, by convention a cube number.
  Size_type a3 =
      (Size_type) ( std::cbrt((Real_type) target_size + 0.5) );

  if ( a3 < 1UL ) { a3 = 1UL ; }

  // One standard intersection is 8 subzone intersections.
  m_n_std_intsc = a3*a3*a3 ;
  m_n_subz_intsc = hexhex_new_npairs_per_std_intsc * m_n_std_intsc ;

  const Int_type block_size = default_gpu_block_size ;
  m_nthreads = hexhex_new_tri_per_pair * m_n_subz_intsc ;  // 72 threads per subzone pair
  m_gsize    = RAJA_DIVIDE_CEILING_INT(m_nthreads, block_size) ;

  setActualProblemSize( m_n_std_intsc ) ;
  setRunReps( target_reps );

  setItsPerRep( m_n_std_intsc );
  setKernelsPerRep(2);   // main intersection kernel and final fixup.

  setBytesAllocatedPerRep( 1*sizeof(Real_type) * 24L*m_n_subz_intsc + // dsubz
                           1*sizeof(Real_type) * 24L*m_n_subz_intsc + // tsubz
                           1*sizeof(Real_type) * hexhex_new_n_vvint_per_block * m_gsize + // vv_int
                           1*sizeof(Real_type) * hexhex_new_nvals_per_pair * m_n_subz_intsc + // vv_out
                           1*sizeof(Real_type) * hexhex_new_nvals_per_pair * m_n_subz_intsc ); // vv

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
  constexpr Size_type flops_per_intsc = flops_per_tri * hexhex_new_tri_per_std_intsc ;

  setFLOPsPerRep(m_n_std_intsc * flops_per_intsc);
}

INTSC_HEXHEX_NEW::~INTSC_HEXHEX_NEW()
{
}


void INTSC_HEXHEX_NEW::setUp(VariantID vid,
                         Size_type RAJAPERF_UNUSED_ARG(tune_idx))
{
  // coordinates for donor zone (the eight corner points)
  Real_type xdzone[8] =
      { m_xmin, m_xmax, m_xmin, m_xmax, m_xmin, m_xmax, m_xmin, m_xmax } ;

  Real_type ydzone[8] =
      { m_ymin, m_ymin, m_ymax, m_ymax, m_ymin, m_ymin, m_ymax, m_ymax } ;

  Real_type zdzone[8] =
      { m_zmin, m_zmin, m_zmin, m_zmin, m_zmax, m_zmax, m_zmax, m_zmax } ;

  Real_type xtzone[8], ytzone[8], ztzone[8] ;
  for ( Index_type i=0 ; i<8 ; ++i ) {
    xtzone[i] = xdzone[i] + m_shift ;
    ytzone[i] = ydzone[i] + m_shift ;
    ztzone[i] = zdzone[i] + m_shift ;
  }

  auto a_ds = allocDataForInit ( m_dsubz, 24L*m_n_subz_intsc, vid ) ;
  auto a_ts = allocDataForInit ( m_tsubz, 24L*m_n_subz_intsc, vid ) ;

  //  Repeat the same calculation m_n_subz_intsc times, expand the
  //  same donor and target zones.
  for ( Index_type k=0 ; k < m_n_subz_intsc ; ++k ) {
    for ( Index_type i=0 ; i<8 ; ++i ) {
      m_dsubz[24L*k+ 0+i] = xdzone[i] ;
      m_dsubz[24L*k+ 8+i] = ydzone[i] ;
      m_dsubz[24L*k+16+i] = zdzone[i] ;
      m_tsubz[24L*k+ 0+i] = xtzone[i] ;
      m_tsubz[24L*k+ 8+i] = ytzone[i] ;
      m_tsubz[24L*k+16+i] = ztzone[i] ;
    }
  }

  // intermediate volumes, moments
  allocData ( m_vv_int, hexhex_new_n_vvint_per_block * m_gsize, vid ) ;

  allocAndInitDataConst ( m_vv_out, hexhex_new_nvals_per_pair * m_n_subz_intsc, 0.0, vid ) ;

  // output volumes and moments on the host
  allocData ( DataSpace::Host, m_vv, hexhex_new_nvals_per_pair * m_n_subz_intsc ) ;
}


//   Number of subzone intersections = 8 * number of standard intersections.
//
void INTSC_HEXHEX_NEW::check_intsc_volume_moments(
    Real_const_ptr vv,   // computed volumes, moments on the host
    VariantID vid)       // Print variant name in case of error
{

  {
    Char_const_ptr tst = "INTSC_HEXHEX_NEW:" ;

    //   Determine the correct volume and moments.
    Real_type v0, vx, vy, vz ;

    Real_type xmin = m_xmin, ymin = m_ymin, zmin = m_zmin ;
    Real_type xmax = m_xmax, ymax = m_ymax, zmax = m_zmax ;

    if ( m_shift > 0.0 ) {
      xmin += m_shift ;   ymin += m_shift ;   zmin += m_shift ;
    } else {
      xmax -= m_shift ;   ymax -= m_shift ;   zmax -= m_shift ;
    }

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

    // Do the check.
    Real_type tolsq = 1.0e-24 ;
    Real_type tolsqv = tolsq * v0*v0 ;
    Real_type tolsqx = tolsq * v0*v0 *
        ( fabs(xmax) + fabs(xmin) ) *  ( fabs(xmax) + fabs(xmin) ) ;
    Real_type tolsqy = tolsq * v0*v0 *
        ( fabs(ymax) + fabs(ymin) ) *  ( fabs(ymax) + fabs(ymin) ) ;
    Real_type tolsqz = tolsq * v0*v0 *
        ( fabs(zmax) + fabs(zmin) ) *  ( fabs(zmax) + fabs(zmin) ) ;

    for ( Index_type k = 0 ; k < m_n_subz_intsc ; ++k ) {

      //  differences between computed and correct
      Real_type dv  = vv[ hexhex_new_nvals_per_pair*k + 0 ] - v0 ;
      Real_type dxm = vv[ hexhex_new_nvals_per_pair*k + 1 ] - vx ;
      Real_type dym = vv[ hexhex_new_nvals_per_pair*k + 2 ] - vy ;
      Real_type dzm = vv[ hexhex_new_nvals_per_pair*k + 3 ] - vz ;

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

        show_comparison ( k, "vv", vv[hexhex_new_nvals_per_pair*k+0], v0, sqrt(tolsqv) ) ;
        show_comparison ( k, "vx", vv[hexhex_new_nvals_per_pair*k+1], vx, sqrt(tolsqx) ) ;
        show_comparison ( k, "vy", vv[hexhex_new_nvals_per_pair*k+2], vy, sqrt(tolsqy) ) ;
        show_comparison ( k, "vz", vv[hexhex_new_nvals_per_pair*k+3], vz, sqrt(tolsqz) ) ;
        getCout() << std::endl ;

        break ;
      }
    }
  }
}


void INTSC_HEXHEX_NEW::updateChecksum(VariantID vid,
                                  size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  copyData ( DataSpace::Host, m_vv,
             getDataSpace(vid), m_vv_out, hexhex_new_nvals_per_pair*m_n_subz_intsc ) ;

  check_intsc_volume_moments ( m_vv, vid ) ;

  addToChecksum(m_vv_out, hexhex_new_nvals_per_pair*m_n_subz_intsc, vid);
}

void INTSC_HEXHEX_NEW::tearDown(VariantID vid,
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
