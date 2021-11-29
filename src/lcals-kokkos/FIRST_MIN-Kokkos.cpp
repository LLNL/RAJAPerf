//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


void FIRST_MIN::runKokkosVariant(VariantID vid)

{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  FIRST_MIN_DATA_SETUP;

//  #define FIRST_MIN_DATA_SETUP \
//  Real_ptr x = m_x;

  auto x_view = getViewFromPointer(x, iend);

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

// https://github.com/kokkos/kokkos/wiki/Kokkos::MinLoc
// MinLoc<T,I,S>::value_type result;
// parallel_reduce(N,Functor,MinLoc<T,I,S>(result));

      Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

               // The 3rd template argument is the memory space in which the
               // result will be stored; the result will be in the place the
               // kernel is called from , i.e., the Host
              using reducer_type = Kokkos::MinLoc<Real_type, Index_type, Kokkos::HostSpace>;
                // must hold the value and the location;
                // Create a variable to hold the result from parallel_reduce
              reducer_type::value_type min_result_obj;

   Kokkos::parallel_reduce("FIRST_MIN_Kokkos Kokkos_Lambda",
                            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                            KOKKOS_LAMBDA(Index_type i, reducer_type::value_type& mymin) {

                                   // #define FIRST_MIN_BODY
                                   // if ( x[i] < mymin.val ) {
                                   //   mymin.val = x[i];
                                   //   mymin.loc = i;
                                   //   }
                                
                                   if (x_view[i] < mymin.val) {
                                        mymin.val = x_view[i];
                                        mymin.loc = i;
                                   }
                                     
                                   // Kokkos can handle a MinLoc type
                                   }, reducer_type(min_result_obj));


        // Kokkos translation of line below 
        // m_minloc = RAJA_MAX(m_minloc, loc.getLoc());
        m_minloc = min_result_obj.loc;

      }
      Kokkos::fence();
      stopTimer();

      break;
    }

    default : {
      std::cout << "\n  FIRST_MIN : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

    moveDataToHostFromKokkosView(x, x_view, iend);
}

} // end namespace lcals
} // end namespace rajaperf
