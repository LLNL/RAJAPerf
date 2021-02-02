//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

struct DaxpyFunctor {
  Real_ptr x;
  Real_ptr y;
  Real_type a;
  DaxpyFunctor(Real_ptr m_x, Real_ptr m_y, Real_type m_a) : DAXPY_FUNCTOR_CONSTRUCT {  }
  void operator()(Index_type i) const { DAXPY_BODY; }
};

void DAXPY::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;

  auto daxpy_lam = [=](Index_type i) {
                     DAXPY_BODY;
                   };

#if defined(RUN_KOKKOS)

  switch ( vid ) {

#if defined(RUN_RAJA_SEQ)
    case Kokkos_Lambda: {
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        Kokkos::parallel_for("DAXPY-KokkosSeq Kokkos_Lambda", Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend),
                             [=](Index_type i) { DAXPY_BODY; });
      }
      stopTimer();
      
      break;
    }
    case Kokkos_Functor_Seq: {
      DaxpyFunctor daxpy_functor_instance(y,x,a);                                
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        Kokkos::parallel_for("DAXPY-KokkosSeq Kokkos_Functor_Seq", Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend),
                             daxpy_functor_instance);
      }
      stopTimer();
      
      break;
    }
#endif // RUN_RAJA_SEQ
    default : {
      std::cout << "\n  DAXPY : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf
