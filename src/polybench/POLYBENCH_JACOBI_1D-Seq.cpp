//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>


namespace rajaperf
{
namespace polybench
{


void POLYBENCH_JACOBI_1D::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto poly_jacobi1d_lam1 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY1;
                            };
  auto poly_jacobi1d_lam2 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY2;
                            };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY1;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
        for (Index_type i = 1; i < N-1; ++i ) {
          POLYBENCH_JACOBI_1D_BODY2;
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
        for (Index_type i = 1; i < N-1; ++i ) {
          poly_jacobi1d_lam1(i);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");
        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
        for (Index_type i = 1; i < N-1; ++i ) {
          poly_jacobi1d_lam2(i);
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_1");
        RAJA::forall<RAJA::seq_exec> ( res,
          RAJA::RangeSegment{1, N-1},
          poly_jacobi1d_lam1
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_1");

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_JACOBI_1D_2");
        RAJA::forall<RAJA::seq_exec> ( res,
          RAJA::RangeSegment{1, N-1},
          poly_jacobi1d_lam2
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_JACOBI_1D_2");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_JACOBI_1D, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace polybench
} // end namespace rajaperf
