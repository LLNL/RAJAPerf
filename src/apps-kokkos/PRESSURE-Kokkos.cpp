//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void PRESSURE::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;

  // Real_ptr compression = m_compression; \
  // Real_ptr bvc = m_bvc; \
  // Real_ptr p_new = m_p_new; \
  // Real_ptr e_old  = m_e_old; \
  // Real_ptr vnewc  = m_vnewc; \

  auto compression_view = getViewFromPointer(compression, iend);
  auto bvc_view = getViewFromPointer(bvc, iend);
  auto p_new_view = getViewFromPointer(p_new, iend);
  auto e_old_view = getViewFromPointer(e_old, iend);
  auto vnewc_view = getViewFromPointer(vnewc, iend);

  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };
  
#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          PRESSURE_BODY2;
        }

      }
      stopTimer();

      break;
    } 

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam1(i);
       }

       for (Index_type i = ibegin; i < iend; ++i ) {
         pressure_lam2(i);
       }

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall<RAJA::loop_exec>(
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

        }); // end sequential region (for single-source code)

      }
      stopTimer(); 

      break;
    }
    */

    case Kokkos_Lambda : {

      Kokkos::fence();
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        //  CRT :  Look at Kokkos graphs as an implementation for kernel
        // seq_region - create a sequential region
        // Intent:  two loop bodies will be executed consecutively
        // https://raja.readthedocs.io/en/v0.9.0/feature/policies.html?highlight=seq_region#parallel-region-policies
        // The sequential region specialization is essentially a pass through operation. 
        // It is provided so that if you want to turn off OpenMP in your code, 
        // you can simply replace the region policy type and you do not have to change your algorithm source code.


          Kokkos::parallel_for("PRESSURE_BODY1 - Kokkos_Lambda",
                          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
                          KOKKOS_LAMBDA(Index_type i) {
                          // #define PRESSURE_BODY1
                          // bvc[i] = cls * (compression[i] + 1.0);
                           bvc_view[i] = cls * (compression_view[i] + 1.0);

                          });



          Kokkos::parallel_for("PRESSURE_BODY2 - Kokkos_Lambda",
                          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
                          KOKKOS_LAMBDA(Index_type i) {
                          // #define PRESSURE_BODY2
                            p_new_view[i] = bvc_view[i] * e_old_view[i] ;
                            if ( fabs(p_new_view[i]) <  p_cut ) p_new_view[i] = 0.0 ;
                            if ( vnewc_view[i] >= eosvmax ) p_new_view[i] = 0.0 ;
                            if ( p_new_view[i]  <  pmin ) p_new_view[i]   = pmin ;
                          });


      }
      Kokkos::fence();
      stopTimer(); 

      break;
    }


    default : {
      std::cout << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

     moveDataToHostFromKokkosView(compression, compression_view, iend);
     moveDataToHostFromKokkosView(bvc, bvc_view, iend);
     moveDataToHostFromKokkosView(p_new, p_new_view, iend);
     moveDataToHostFromKokkosView(e_old, e_old_view, iend);
     moveDataToHostFromKokkosView(vnewc, vnewc_view, iend);

}

} // end namespace apps
} // end namespace rajaperf
