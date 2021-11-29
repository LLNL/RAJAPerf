//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

  void MUL::runKokkosVariant(VariantID vid) 
  {
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();


  MUL_DATA_SETUP;

  /* from MUL.hpp
  #define MUL_DATA_SETUP \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_type alpha = m_alpha

*/
  auto b_view = getViewFromPointer(b, iend);
  auto c_view = getViewFromPointer(c, iend);
 

  auto mul_lam = [=](Index_type i) {
                   MUL_BODY;
                 };


#if defined(RUN_KOKKOS)

  switch ( vid ) {

  case Kokkos_Lambda : {

        Kokkos::fence();
        startTimer();

        for (RepIndex_type irep =0; irep < run_reps; ++irep) {
                
                Kokkos::parallel_for("MUL_Kokkos Kokkos_Lambda",
                                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
                                KOKKOS_LAMBDA(Index_type i) {
                                // MUL BODY DEFINITION:
                                // b[i] = alpha * c[i] ;
                                b_view[i] = alpha * c_view[i];
                                });

        }
        Kokkos::fence();
        stopTimer();

        break;

        }

    default : {
      std::cout << "\n  MUL : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(b, b_view, iend);
  moveDataToHostFromKokkosView(c, c_view, iend);


}

} // end namespace stream
} // end namespace rajaperf
