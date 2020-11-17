//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{
struct AtomicPIFunctor {
  Real_type dx;
  Real_ptr pi;

  AtomicPIFunctor(Real_type m_dx, Real_ptr m_pi) : ATOMIC_PI_FUNCTOR_CONSTRUCT {}
};


void ATOMIC_PI::runKokkosOpenMPVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

#if defined(RUN_KOKKOS) && defined(RUN_OPENMP)
  switch ( vid ) {

    case Kokkos_Functor_OpenMP : {

      startTimer();
      //for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //  *pi = m_pi_init;
      //  RAJA::forall<RAJA::omp_parallel_for_exec>( 
      //    RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
      //      double x = (double(i) + 0.5) * dx;
      //      RAJA::atomicAdd<RAJA::omp_atomic>(pi, dx / (1.0 + x * x));
      //  });
      //  *pi *= 4.0;

      //}
      stopTimer();

      break;
    }
    case Kokkos_Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        *pi = m_pi_init;

        Kokkos::parallel_for("name",Kokkos::RangePolicy<Kokkos::OpenMP>(ibegin, iend), KOKKOS_LAMBDA(Index_type i){
          double x = ((double(i) + 0.5) * dx);
          Kokkos::atomic_add(pi, dx / (1.0 + x * x));
	});
        *pi *= 4.0;
      }
      stopTimer();

      break;
    }


    default : {
      std::cout << "\n  ATOMIC_PI : Unknown variant id = " << vid << std::endl;
    }

  }

#endif
}

} // end namespace basic
} // end namespace rajaperf
