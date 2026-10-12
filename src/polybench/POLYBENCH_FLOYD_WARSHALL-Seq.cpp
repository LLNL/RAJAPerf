//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{


void POLYBENCH_FLOYD_WARSHALL::runSeqVariant(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        for (Index_type k = 0; k < N; ++k) {
          for (Index_type i = 0; i < N; ++i) {
            for (Index_type j = 0; j < N; ++j) {
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto poly_floydwarshall_base_lam = [=](Index_type k, Index_type i,
                                             Index_type j) {
                                           POLYBENCH_FLOYD_WARSHALL_BODY;
                                         };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        for (Index_type k = 0; k < N; ++k) {
          for (Index_type i = 0; i < N; ++i) {
            for (Index_type j = 0; j < N; ++j) {
              poly_floydwarshall_base_lam(k, i, j);
            }
          }
        }
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

      auto poly_floydwarshall_lam = [=](Index_type k, Index_type i,
                                        Index_type j) {
                                      POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
                                    };

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::seq_exec,
            RAJA::statement::For<1, RAJA::seq_exec,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N},
                           RAJA::RangeSegment{0, N}),
          res,
          poly_floydwarshall_lam
        );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FLOYD_WARSHALL, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace polybench
} // end namespace rajaperf
