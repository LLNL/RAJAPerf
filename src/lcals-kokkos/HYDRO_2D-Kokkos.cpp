//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void HYDRO_2D::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  // Wrap input pointers in Kokkos::Views
/*
#define HYDRO_2D_DATA_SETUP \
  Real_ptr zadat = m_za; \
  Real_ptr zbdat = m_zb; \
  Real_ptr zmdat = m_zm; \
  Real_ptr zpdat = m_zp; \
  Real_ptr zqdat = m_zq; \
  Real_ptr zrdat = m_zr; \
  Real_ptr zudat = m_zu; \
  Real_ptr zvdat = m_zv; \
  Real_ptr zzdat = m_zz; \
\
  Real_ptr zroutdat = m_zrout; \
  Real_ptr zzoutdat = m_zzout; \
\

*/
// ATTN:  THESE INPUTS ARE 2D Views:
//
    auto zadat_view = getViewFromPointer(zadat, kn, jn );
    auto zbdat_view = getViewFromPointer(zbdat, kn, jn );
    auto zmdat_view = getViewFromPointer(zmdat, kn, jn );
    auto zpdat_view = getViewFromPointer(zpdat, kn, jn );
    auto zqdat_view = getViewFromPointer(zqdat, kn, jn );
    auto zrdat_view = getViewFromPointer(zrdat, kn, jn );
    auto zudat_view = getViewFromPointer(zudat, kn, jn );
    auto zvdat_view = getViewFromPointer(zvdat, kn, jn );
    auto zzdat_view = getViewFromPointer(zzdat, kn, jn );

    // Wrap output pointers into Kokkos::Views

    auto zroutdat_view = getViewFromPointer(zroutdat, kn, jn );
    auto zzoutdat_view = getViewFromPointer(zzoutdat, kn, jn );

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {


         // Use MDRangePolicy for multidimensional arrays
         // https://github.com/kokkos/kokkos/wiki/Kokkos::MDRangePolicy

        Kokkos::parallel_for("HYDRO_2D_Kokkos Kokkos_Lambda--BODY1",
                             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({kbeg,jbeg}, {kend,jend}),
                             KOKKOS_LAMBDA(int64_t k, int64_t j) {
                              /*                             
                              #define HYDRO_2D_BODY1_RAJA  \
                              za(k,j) = ( zp(k+1,j-1) + zq(k+1,j-1) - zp(k,j-1) - zq(k,j-1) ) * \
                              ( zr(k,j) + zr(k,j-1) ) / ( zm(k,j-1) + zm(k+1,j-1) ); \
                              zb(k,j) = ( zp(k,j-1) + zq(k,j-1) - zp(k,j) - zq(k,j) ) * \
                              ( zr(k,j) + zr(k-1,j) ) / ( zm(k,j) + zm(k,j-1));
                              */
                              zadat_view(k,j) = ( zpdat_view(k+1,j-1) + zqdat_view(k+1,j-1) - zpdat_view(k,j-1) - zqdat_view(k,j-1) ) * \
                              ( zrdat_view(k,j) + zrdat_view(k,j-1) ) / ( zmdat_view(k,j-1) + zmdat_view(k+1,j-1) ); \

                              zbdat_view(k,j) = ( zpdat_view(k,j-1) + zqdat_view(k,j-1) - zpdat_view(k,j) - zqdat_view(k,j) ) * \
                              ( zrdat_view(k,j) + zrdat_view(k-1,j) ) / ( zmdat_view(k,j) + zmdat_view(k,j-1));
                             });


        Kokkos::parallel_for("HYDRO_2D_Kokkos Kokkos_Lambda--BODY2",
                             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({kbeg,jbeg}, {kend,jend}),
                             KOKKOS_LAMBDA(int64_t k, int64_t j) {

								/*
								 #define HYDRO_2D_BODY2_RAJA \
								 zu(k,j) += s*( za(k,j) * ( zz(k,j) - zz(k,j+1) ) - \
								 za(k,j-1) * ( zz(k,j) - zz(k,j-1) ) - \
								 zb(k,j) * ( zz(k,j) - zz(k-1,j) ) + \
								 zb(k+1,j) * ( zz(k,j) - zz(k+1,j) ) ); \
								 zv(k,j) += s*( za(k,j) * ( zr(k,j) - zr(k,j+1) ) - \
								 za(k,j-1) * ( zr(k,j) - zr(k,j-1) ) - \
								 zb(k,j) * ( zr(k,j) - zr(k-1,j) ) + \
								 zb(k+1,j) * ( zr(k,j) - zr(k+1,j) ) );
								*/

								 zudat_view(k,j) += s*( zadat_view(k,j) * ( zzdat_view(k,j) - zzdat_view(k,j+1) ) - \
								 zadat_view(k,j-1) * (zzdat_view(k,j) - zzdat_view(k,j-1) ) - \
								 zbdat_view(k,j) * ( zzdat_view(k,j) - zzdat_view(k-1,j) ) + \
								 zbdat_view(k+1,j) * ( zzdat_view(k,j) - zzdat_view(k+1,j) ) ); \
								 zvdat_view(k,j) += s*( zadat_view(k,j) * ( zrdat_view(k,j) - zrdat_view(k,j+1) ) - \
								 zadat_view(k,j-1) * ( zrdat_view(k,j) - zrdat_view(k,j-1) ) - \
								 zbdat_view(k,j) * ( zrdat_view(k,j) - zrdat_view(k-1,j) ) + \
								 zbdat_view(k+1,j) * ( zrdat_view(k,j) - zrdat_view(k+1,j) ) );

                             });


        Kokkos::parallel_for("HYDRO_2D_Kokkos Kokkos_Lambda--BODY3",
                             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({kbeg,jbeg}, {kend,jend}),
                             KOKKOS_LAMBDA(int64_t k, int64_t j) {
                             /*
                             #define HYDRO_2D_BODY3_RAJA \
                             zrout(k,j) = zr(k,j) + t*zu(k,j); \
                             zzout(k,j) = zz(k,j) + t*zv(k,j);
                             */
                             
                             zroutdat_view(k,j) = zrdat_view(k,j) + t*zudat_view(k,j); \
                             zzoutdat_view(k,j) = zzdat_view(k,j) + t*zvdat_view(k,j);
                             });

      }
      
      Kokkos::fence();
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS



  // Wrap input pointers in Kokkos::Views
/*
#define HYDRO_2D_DATA_SETUP \
  Real_ptr zadat = m_za; \
  Real_ptr zbdat = m_zb; \
  Real_ptr zmdat = m_zm; \
  Real_ptr zpdat = m_zp; \
  Real_ptr zqdat = m_zq; \
  Real_ptr zrdat = m_zr; \
  Real_ptr zudat = m_zu; \
  Real_ptr zvdat = m_zv; \
  Real_ptr zzdat = m_zz; \
\
  Real_ptr zroutdat = m_zrout; \
  Real_ptr zzoutdat = m_zzout; \
\

*/

  // There are 9 input views: 
  moveDataToHostFromKokkosView(zadat, zadat_view, kn, jn);
  moveDataToHostFromKokkosView(zbdat, zbdat_view, kn, jn);
  moveDataToHostFromKokkosView(zmdat, zmdat_view, kn, jn);
  moveDataToHostFromKokkosView(zpdat, zpdat_view, kn, jn);
  moveDataToHostFromKokkosView(zqdat, zqdat_view, kn, jn);
  moveDataToHostFromKokkosView(zrdat, zrdat_view, kn, jn);
  moveDataToHostFromKokkosView(zudat, zudat_view, kn, jn);
  moveDataToHostFromKokkosView(zvdat, zvdat_view, kn, jn);
  moveDataToHostFromKokkosView(zzdat, zzdat_view, kn, jn);
  
  // There are 2 output views:
  moveDataToHostFromKokkosView(zroutdat, zroutdat_view, kn, jn);
  moveDataToHostFromKokkosView(zzoutdat, zzoutdat_view, kn, jn);

}

} // end namespace lcals
} // end namespace rajaperf
