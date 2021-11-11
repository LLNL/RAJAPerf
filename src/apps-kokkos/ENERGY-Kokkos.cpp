//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void ENERGY::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ENERGY_DATA_SETUP;

  // Instantiate Kokkos::Views
    //auto a_view = getViewFromPointer(a, iend);

	auto e_new_view = getViewFromPointer(e_new, iend);
	auto e_old_view = getViewFromPointer(e_old, iend);
	auto delvc_view = getViewFromPointer(delvc, iend);
	auto p_new_view = getViewFromPointer(p_new, iend);
	auto p_old_view = getViewFromPointer(p_old, iend);
	auto q_new_view = getViewFromPointer(q_new, iend);
	auto q_old_view = getViewFromPointer(q_old, iend);
	auto work_view = getViewFromPointer(work, iend);
	auto compHalfStep_view = getViewFromPointer(compHalfStep, iend);
	auto pHalfStep_view = getViewFromPointer(pHalfStep, iend);
	auto bvc_view = getViewFromPointer(bvc, iend);
	auto pbvc_view = getViewFromPointer(pbvc, iend);
	auto ql_old_view = getViewFromPointer(ql_old, iend);
	auto qq_old_view = getViewFromPointer(qq_old, iend);
	auto vnewc_view = getViewFromPointer(vnewc, iend);

  
  auto energy_lam1 = [=](Index_type i) {
                       ENERGY_BODY1;
                     };
  auto energy_lam2 = [=](Index_type i) {
                       ENERGY_BODY2;
                     };
  auto energy_lam3 = [=](Index_type i) {
                       ENERGY_BODY3;
                     };
  auto energy_lam4 = [=](Index_type i) {
                       ENERGY_BODY4;
                     };
  auto energy_lam5 = [=](Index_type i) {
                       ENERGY_BODY5;
                     };
  auto energy_lam6 = [=](Index_type i) {
                       ENERGY_BODY6;
                     };

#if defined(RUN_KOKKOS)
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }
  
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }

      }
      stopTimer();

      break;
    } 

    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam1(i);
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam2(i);
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam3(i);
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam4(i);
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam5(i);
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          energy_lam6(i);
        }

      }
      stopTimer();

      break;
    }

    case Kokkos_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

              Kokkos::parallel_for("ENERGY - lambda 1",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              // Lamda Body 1
                                e_new_view[i] = e_old_view[i] - 0.5 * delvc_view[i] * \
                                (p_old_view[i] + ql_old_view[i]) + 0.5 * work_view[i];

                              });

              Kokkos::parallel_for("ENERGY - lambda 2",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              //#define ENERGY_BODY2
                          if ( delvc_view[i] > 0.0 ) {
                             q_new_view[i] = 0.0 ;
                          } \
                          else { \
                             Real_type vhalf = 1.0 / (1.0 + compHalfStep_view[i]) ; 
                             Real_type ssc = ( pbvc[i] * e_new_view[i]
                                + vhalf * vhalf * bvc[i] * pHalfStep_view[i] ) / rho0 ;
                             if ( ssc <= 0.1111111e-36 ) {
                                ssc = 0.3333333e-18 ;
                             } else { 
                                ssc = sqrt(ssc) ; 
                             }
                             q_new_view[i] = (ssc*ql_old_view[i] + qq_old_view[i]) ;
                          }
                              });


              Kokkos::parallel_for("ENERGY - lambda 3",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              //#define ENERGY_BODY3
                              
                          e_new_view[i] = e_new_view[i] + 0.5 * delvc_view[i] \
                                     * ( 3.0*(p_old_view[i] + qq_old_view[i]) \
                                         - 4.0*(pHalfStep_view[i] + q_new_view[i])) ;


                          });



              Kokkos::parallel_for("ENERGY - lambda 4",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              //#define ENERGY_BODY4
                              
                                  e_new_view[i] += 0.5 * work_view[i]; \
  if ( fabs(e_new_view[i]) < e_cut ) { e_new_view[i] = 0.0  ; } \
  if ( e_new_view[i]  < emin ) { e_new_view[i] = emin ; }

                          });


              Kokkos::parallel_for("ENERGY - lambda 5",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              //#define ENERGY_BODY5
  Real_type q_tilde ; \

  if (delvc_view[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc_view[i] * e_new_view[i] \
         + vnewc_view[i] * vnewc_view[i] * bvc_view[i] * p_new_view[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old_view[i]) ; \
  } \
  e_new_view[i] = e_new_view[i] - ( 7.0*(p_old_view[i] + q_old_view[i]) \
                         - 8.0*(pHalfStep_view[i] + q_new_view[i]) \
                         + (p_new_view[i] + q_tilde)) * delvc_view[i] / 6.0 ; \
  if ( fabs(e_new_view[i]) < e_cut ) { \
     e_new_view[i] = 0.0  ; \
  } \
  if ( e_new_view[i]  < emin ) { \
     e_new_view[i] = emin ; \
  }
                             

                          });


              Kokkos::parallel_for("ENERGY - lambda 6",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                              KOKKOS_LAMBDA(const int64_t i){
                              //#define ENERGY_BODY6
                              
  if ( delvc_view[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc_view[i] * e_new_view[i] \
             + vnewc_view[i] * vnewc_view[i] * bvc_view[i] * p_new_view[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new_view[i] = (ssc*ql_old_view[i] + qq_old_view[i]) ; \
     if (fabs(q_new_view[i]) < q_cut) q_new_view[i] = 0.0 ; \
                          }


        }); 

      }
      stopTimer(); 

      break;
    }


    default : {
      std::cout << "\n  ENERGY : Unknown variant id = " << vid << std::endl;
    }

  }


#endif // RUN_KOKKOS

   //moveDataToHostFromKokkosView(a, a_view, iend);
   moveDataToHostFromKokkosView(e_new, e_new_view, iend);
   moveDataToHostFromKokkosView(e_old, e_old_view, iend);
   moveDataToHostFromKokkosView(delvc, delvc_view, iend);
   moveDataToHostFromKokkosView(p_new, p_new_view, iend);
   moveDataToHostFromKokkosView(p_old, p_old_view, iend);
   moveDataToHostFromKokkosView(q_new, q_new_view, iend);
   moveDataToHostFromKokkosView(q_old, ql_old_view, iend);
   moveDataToHostFromKokkosView(work, work_view, iend);
   moveDataToHostFromKokkosView(compHalfStep, compHalfStep_view, iend);
   moveDataToHostFromKokkosView(pHalfStep, pHalfStep_view, iend);
   moveDataToHostFromKokkosView(bvc, bvc_view, iend);
   moveDataToHostFromKokkosView(pbvc, pbvc_view, iend);
   moveDataToHostFromKokkosView(ql_old, ql_old_view, iend);
   moveDataToHostFromKokkosView(qq_old, qq_old_view, iend);
   moveDataToHostFromKokkosView(vnewc, vnewc_view, iend);

}

} // end namespace apps
} // end namespace rajaperf
