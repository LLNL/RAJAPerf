//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


// Kokkos-ify here

void IF_QUAD::runKokkosSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  IF_QUAD_DATA_SETUP;

  auto ifquad_lam = [=](Index_type i) {
                      IF_QUAD_BODY;
                    };


#if defined(RUN_KOKKOS)

  switch ( vid ) {



#if defined(RUN_RAJA_SEQ)     

    case Kokkos_Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

/*        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), ifquad_lam);
*/
	// Translation 
	Kokkos::parallel_for("IF_QUAD_KokkosSeq Kokkos_Lambda_Seq", Kokkos::RangePolicy<Kokkos::Serial>(ibegin, iend),

		[=] (Index_type i) {IF_QUAD_BODY});

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  IF_QUAD : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS


}

} // end namespace basic
} // end namespace rajaperf
