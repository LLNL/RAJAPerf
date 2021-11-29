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


void DOT::runKokkosVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  // Instantiation of pointer - wrapped Kokkos views:
  auto a_view = getViewFromPointer(a, iend);
  auto b_view = getViewFromPointer(b, iend);

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {
                          Kokkos::fence();
                          startTimer();

                          for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

                                   Real_type dot = m_dot_init;

                                   parallel_reduce("DOT-Kokkos Kokkos_Lambda",
                                                  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                                  KOKKOS_LAMBDA(const int64_t i, Real_type& dot_res){

                                                  // DOT BODY definition from header:
                                                  //   dot += a[i] * b[i] ;
                                                  dot_res += a_view[i]*b_view[i];
                                                  }, dot);
                                  m_dot += static_cast<Real_type>(dot);
                          }

                          Kokkos::fence();
                          stopTimer();
                          
                          break;
                         }


    default : {
      std::cout << "\n  DOT : Unknown variant id = " << vid << std::endl;
    }

  }


#endif // RUN_KOKKOS
              
        moveDataToHostFromKokkosView(a, a_view, iend);
        moveDataToHostFromKokkosView(b, b_view, iend);

}

} // end namespace stream
} // end namespace rajaperf
