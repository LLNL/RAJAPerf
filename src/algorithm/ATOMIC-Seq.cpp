//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


template < size_t replication >
void ATOMIC::runSeqVariantReplicate(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  ATOMIC_DATA_SETUP(replication);

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ATOMIC_BODY(i, ATOMIC_VALUE);
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto atomic_base_lam = [=](Index_type i) {
                                 ATOMIC_BODY(i, ATOMIC_VALUE);
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          atomic_base_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::seq_exec>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            ATOMIC_RAJA_BODY(RAJA::seq_atomic, i, ATOMIC_VALUE);
        });

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  ATOMIC : Unknown variant id = " << vid << std::endl;
    }

  }

  ATOMIC_DATA_TEARDOWN(replication);

}


void ATOMIC::runSeqVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_Seq || vid == Lambda_Seq || vid == RAJA_Seq ) {

    seq_for(cpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        if (tune_idx == t) {

          runSeqVariantReplicate<replication>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  ATOMIC : Unknown OMP Target variant id = " << vid << std::endl;

  }

}

void ATOMIC::setSeqTuningDefinitions(VariantID vid)
{
  if ( vid == Base_Seq || vid == Lambda_Seq || vid == RAJA_Seq ) {

    seq_for(cpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        addVariantTuningName(vid, "replicate_"+std::to_string(replication));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf
