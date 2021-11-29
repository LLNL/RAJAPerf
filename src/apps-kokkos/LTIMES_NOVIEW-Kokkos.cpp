//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


void LTIMES_NOVIEW::runKokkosVariant(VariantID vid)
{
        // Nota bene: we put a return statement for build and running purposes;
        // A kernel without a Kokkos view is not informative for Kokkos
        // performance
        return;
  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;
 
  auto ltimesnoview_lam = [=](Index_type d, Index_type z, 
                              Index_type g, Index_type m) {
                                LTIMES_NOVIEW_BODY;
                          };

#if defined (RUN_KOKKOS)

  switch ( vid ) {

/*
    Future Kokkos Translation here:
    case Kokkos_Lambda : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<1, RAJA::loop_exec,       // z
            RAJA::statement::For<2, RAJA::loop_exec,     // g
              RAJA::statement::For<3, RAJA::loop_exec,   // m
                RAJA::statement::For<0, RAJA::loop_exec, // d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                                                 RAJA::RangeSegment(0, num_z),
                                                 RAJA::RangeSegment(0, num_g),
                                                 RAJA::RangeSegment(0, num_m)),
                                ltimesnoview_lam
                              );

      }
      stopTimer(); 

      break;

    }

*/
    default : {
      std::cout << "\n LTIMES_NOVIEW : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS 

// Move data from Kokkos View on device back to the host


}

} // end namespace apps
} // end namespace rajaperf
