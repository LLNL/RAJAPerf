//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SOLVE_4x4_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void SOLVE_4x4_4GROUPS::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  std::cout.precision(17);


  auto solve_4x4_4g_lam = [=](Index_type i) {
   
      Real_type * RAJA_RESTRICT  A = m_a + 16*4 * i;
      Real_type * RAJA_RESTRICT  x = m_x +  4*4 * i;
      Real_type * RAJA_RESTRICT  y = m_y +  4*4 * i;

      using vec_t = RAJA::expt::VectorRegister<Real_type>;
      using idx_t = RAJA::expt::VectorIndex<int, vec_t>;

      using Vec = RAJA::View<Real_type, RAJA::StaticLayout<RAJA::PERM_I, 4>>; 
      auto vall = idx_t::static_all();

      auto a00 = Vec  ( A + 0 );
      auto a10 = Vec  ( A + 4 );
      auto a20 = Vec  ( A + 8 );
      auto a30 = Vec  ( A +12 );
      auto a01 = Vec  ( A +16 );
      auto a11 = Vec  ( A +20 );
      auto a21 = Vec  ( A +24 );
      auto a31 = Vec  ( A +28 );
      auto a02 = Vec  ( A +32 );
      auto a12 = Vec  ( A +36 );
      auto a22 = Vec  ( A +40 );
      auto a32 = Vec  ( A +44 );
      auto a03 = Vec  ( A +48 );
      auto a13 = Vec  ( A +52 );
      auto a23 = Vec  ( A +56 );
      auto a33 = Vec  ( A +60 );

      Real_type L[4*6];
      auto l10 = Vec  ( L + 0 );
      auto l20 = Vec  ( L + 4 );
      auto l30 = Vec  ( L + 8 );
      auto l21 = Vec  ( L +12 );
      auto l31 = Vec  ( L +16 );
      auto l32 = Vec  ( L +20 );

      Real_type U[4*10];
      auto u00 = Vec  ( U + 0 );
      auto u01 = Vec  ( U + 4 );
      auto u02 = Vec  ( U + 8 );
      auto u03 = Vec  ( U +12 );
      auto u11 = Vec  ( U +16 );
      auto u12 = Vec  ( U +20 );
      auto u13 = Vec  ( U +24 );
      auto u22 = Vec  ( U +28 );
      auto u23 = Vec  ( U +32 );
      auto u33 = Vec  ( U +36 );


      auto x0 =  Vec  ( x + 0 );
      auto x1 =  Vec  ( x + 4 );
      auto x2 =  Vec  ( x + 8 );
      auto x3 =  Vec  ( x +12 );

      auto y0 =  Vec  ( y + 0 );
      auto y1 =  Vec  ( y + 4 );
      auto y2 =  Vec  ( y + 8 );
      auto y3 =  Vec  ( y +12 );

      Real_type Z[4*4];  
      auto z0 =  Vec  ( Z + 0 );
      auto z1 =  Vec  ( Z + 4 );
      auto z2 =  Vec  ( Z + 8 );
      auto z3 =  Vec  ( Z +12 );

      Real_type R[4*4];  
      auto r0 =  Vec  ( R + 0  );
      auto r1 =  Vec  ( R + 4  );
      auto r2 =  Vec  ( R + 8  );
      auto r3 =  Vec  ( R +12  );



      u00(vall) = a00(vall);
      u01(vall) = a01(vall);
      u02(vall) = a02(vall);
      u03(vall) = a03(vall);

      l10(vall) = a10(vall) / a00(vall);
      l20(vall) = a20(vall) / a00(vall);
      l30(vall) = a30(vall) / a00(vall);

      // used for intermediate calculations
      u11(vall) = a11(vall) - l10(vall) * u01(vall);
      u12(vall) = a12(vall) - l10(vall) * u02(vall);
      u13(vall) = a13(vall) - l10(vall) * u03(vall);
      // column
      l21(vall)  = (a21(vall) - l20(vall) * u01(vall)) /  u11(vall);
      l31(vall) = (a31(vall) - l30(vall) * u01(vall)) /  u11(vall);

      // k == 2
      // row
      u22(vall)   = a22(vall) - l20(vall) * u02(vall) - l21(vall) * u12(vall);
      u23(vall)   = a23(vall) - l20(vall) * u03(vall) - l21(vall) * u13(vall);
      // column 
      l32(vall)  = (a32(vall) - l30(vall) * u02(vall) - l31(vall) * u12(vall)) /u22(vall);

      // k == 3
      // row
      u33(vall)  = a33(vall) - l30(vall) * u03(vall) - l31(vall) * u13(vall) -  l32(vall) * u23(vall);


      // Now solve

      // Load right-hand side in to fast memory
      r0(vall) = x0(vall);
      r1(vall) = x1(vall);
      r2(vall) = x2(vall);
      r3(vall) = x3(vall);

      // Forward substitution
      z0(vall)  =  r0(vall);
      z1(vall)  = (r1(vall) - l10(vall) * z0(vall));
      z2(vall)  = r2(vall) - l20(vall) * z0(vall) - l21(vall) * z1(vall);
      z3(vall)  = r3(vall) - l30(vall) * z0(vall) - l31(vall) * z1(vall) - l32(vall) * z2(vall);

      // Backward substitution
      r3(vall)  = z3(vall) / u33(vall);
      r2(vall)  = (z2(vall) - u23(vall) * r3(vall)) / u22(vall);
      r1(vall)  = (z1(vall) - u12(vall) * r2(vall) - u13(vall) * r3(vall)) /  u11(vall);
      r0(vall)  = (z0(vall) - u01(vall) * r1(vall) - u02(vall) * r2(vall) -  u03(vall) * r3(vall) ) / u00(vall);

      y0(vall) = r0(vall);
      y1(vall) = r1(vall);
      y2(vall) = r2(vall);
      y3(vall) = r3(vall);
  };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {

            Real_type* RAJA_RESTRICT  a = m_a+16*4*i;
            Real_type* RAJA_RESTRICT  x = m_x+ 4*4*i;
            Real_type* RAJA_RESTRICT  y = m_y+ 4*4*i;

            Real_type l10, l20, l30,
                           l21, l31,
                                l32;

            Real_type u00, u01, u02, u03,
                           u11, u12, u13,
                                u22, u23,
                                     u33;

            Real_type z0, z1, z2, z3; 
            Real_type r0, r1, r2, r3;  

            for( int j=0; j<4; j++) {

                u00 = a[(0*4+0)*4+j];
                u01 = a[(1*4+0)*4+j];
                u02 = a[(2*4+0)*4+j];
                u03 = a[(3*4+0)*4+j];


                l10 = a[(0*4+1)*4+j] / a[(0*4+0)*4+j];
                l20 = a[(0*4+2)*4+j] / a[(0*4+0)*4+j];
                l30 = a[(0*4+3)*4+j] / a[(0*4+0)*4+j];

                // used for intermediate calculations
                u11 = a[(1*4+1)*4+j] - l10 * u01;
                u12 = a[(2*4+1)*4+j] - l10 * u02;
                u13 = a[(3*4+1)*4+j] - l10 * u03;
                // column
                l21  = (a[(1*4+2)*4+j] - l20 * u01);
                l21 /= u11;
                l31  = (a[(1*4+3)*4+j] - l30 * u01);
                l31 /= u11;

                // k == 2
                // row
                u22  = a[(2*4+2)*4+j] - l20 * u02 -  l21 * u12;
                u23  = a[(3*4+2)*4+j] - l20 * u03 -  l21 * u13;
                // column
                l32  = a[(2*4+3)*4+j] - l30 * u02 - l31 * u12;
                l32 /=u22;

                // k == 3
                // row
                u33  = a[(3*4+3)*4+j] - l30 * u03 - l31 * u13 - l32 * u23;


                // Now solve

                // Load right-hand side in to fast memory
                r0 = x[0*4+j];
                r1 = x[1*4+j];
                r2 = x[2*4+j];
                r3 = x[3*4+j];

                // Forward substitution
                z0  =  r0;
                z1  = (r1 - l10 * z0);
                z2  = r2 - l20 * z0 - l21 * z1;
                z3  = r3 - l30 * z0 - l31 * z1 - l32 * z2;

                // Backward substitution
                r3  = z3/u33;
                r2  = (z2 - u23 * r3);
                r2 /= u22;
                r1  = z1 - u12 * r2 - u13 * r3;
                r1 /= u11;
                r0  = z0 - u01 * r1 - u02 * r2 - u03 * r3;
                r0 /= u00;

                y[0*4+j] = r0;
                y[1*4+j] = r1;
                y[2*4+j] = r2;
                y[3*4+j] = r3;

            }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          solve_4x4_4g_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), solve_4x4_4g_lam);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  SOLVE_4x4_4GROUPS : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
