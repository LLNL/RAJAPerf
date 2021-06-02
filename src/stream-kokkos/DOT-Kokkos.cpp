//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

/*
void DOT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();
*/


void DOT::runKokkosVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();



  DOT_DATA_SETUP;

  // Instantiation of pointer - wrapped views:
  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);
  //
  // From basic-kokkos - REDUCE3 
  // Instantiation of a view from a pointer to a vector
  // auto vec_view = getViewFromPointer(vec, iend);



  // Pre-processor directive 
#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          DOT_BODY;
        }

         m_dot += dot;

      }
      stopTimer();

      break;
    }

// #if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto dot_base_lam = [=](Index_type i) -> Real_type {
                            return a[i] * b[i];
                          };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          dot += dot_base_lam(i);
        }

        m_dot += dot;

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Real_type> dot(m_dot_init);

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          DOT_BODY;
        });

        m_dot += static_cast<Real_type>(dot.get());

      }
      stopTimer();

      break;
    }
    */

    case Kokkos_Lambda : {
                          
                          // open Kokkosfence
                          Kokkos::fence();
                          startTimer();

                          for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
                                   // Declare and initialize dot
                                   // dot will contain the reduction value,
                                   // i.e., the dot product
                                   //
                                   // Reductions combine contributions from
                                   // loop iterations
                                   Real_type dot = m_dot_init;

                                   parallel_reduce("DOT-Kokkos Kokkos_Lambda",
                                                  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                                  KOKKOS_LAMBDA(const int64_t i, Real_type& dot_res){

                                                  // DOT BODY definition from header:
                                                  //   dot += a[i] * b[i] ;
                                                  //dot_res += a_view[i]*b_view[i];
                                                  ///////////////////////////////
                                                  //Int_type vec_i = vec_view[i];
                                                  dot_res += a_view[i]*b_view[i];
                                                  //dot_res = vec_i;
                                                  }, dot);
                                  m_dot += static_cast<Real_type>(dot);
                          }

                          Kokkos::fence();
                          stopTimer();
                          
                          break;
                         }



// #endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }


#endif // RUN_KOKKOS
              
         std::cout << " FIX ME STREAM DOT -- GET DATA FROM VIEWS " << std::endl;
        //moveDataToHostFromKokkosView(a, a_view, iend);
        //moveDataToHostFromKokkosView(b, b_view, iend);
        
        // From REDUCE3-INT
        // moveDataToHostFromKokkosView(vec, vec_view, iend);

}

} // end namespace stream
} // end namespace rajaperf
