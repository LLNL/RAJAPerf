//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;

template < size_t replication >
void ATOMIC::runOpenMPTargetVariantReplicate(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      #pragma omp target is_device_ptr(atomic)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_OMP, i, ATOMIC_VALUE);
      }
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    auto res{getOmpTargetResource()};

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("ATOMIC_1");
      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>( res,
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ATOMIC_BODY(RAJAPERF_ATOMIC_ADD_RAJA_OMP, i, ATOMIC_VALUE);
      });
      RP_CALI_SUBKERNEL_END("ATOMIC_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown OMP Target variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);

}


void ATOMIC::defineOpenMPTargetVariantTunings()
{

  for (VariantID vid : {Base_OpenMPTarget, RAJA_OpenMPTarget}) {

    seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        addVariantTuning<&ATOMIC::runOpenMPTargetVariantReplicate<replication>>(
            vid, "replicate_"+std::to_string(replication));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
