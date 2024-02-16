//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
void ATOMIC::runOpenMPTargetReplicate(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #pragma omp target is_device_ptr(atomic)
      #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
      for (Index_type i = ibegin; i < iend; ++i ) {
        #pragma omp atomic
        ATOMIC_BODY(i, ATOMIC_VALUE);
      }

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
        RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ATOMIC_RAJA_BODY(RAJA::omp_atomic, i, ATOMIC_VALUE);
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown OMP Target variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);

}

void ATOMIC::runOpenMPTargetVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_OpenMPTarget || vid == RAJA_OpenMPTarget ) {

    seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        if (tune_idx == t) {

          runOpenMPTargetVariantReplicate<replication>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  ATOMIC : Unknown OMP Target variant id = " << vid << std::endl;

  }

}

void ATOMIC::setOpenMPTargetTuningDefinitions(VariantID vid)
{
  if ( vid == Base_OpenMPTarget || vid == RAJA_OpenMPTarget ) {

    seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        addVariantTuningName(vid, "replicate_"+std::to_string(replication));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
