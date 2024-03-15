//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{


void REDUCE_SUM::runSeqVariantDefault(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Data_type sum = m_sum_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_SUM_BODY;
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      auto reduce_sum_base_lam = [=](Index_type i) {
                                 return x[i];
                               };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Data_type sum = m_sum_init;

        for (Index_type i = ibegin; i < iend; ++i ) {
          sum += reduce_sum_base_lam(i);
        }

        m_sum = sum;

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::ReduceSum<RAJA::seq_reduce, Data_type> sum(m_sum_init);

        RAJA::forall<RAJA::seq_exec>( RAJA::RangeSegment(ibegin, iend),
          [=](Index_type i) {
            REDUCE_SUM_BODY;
        });

        m_sum = sum.get();

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

}


template < size_t replication >
void REDUCE_SUM::runSeqVariantReplication(VariantID vid)
{
  static_assert(replication == detail::next_pow2(replication), "");

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Data_type rep_sum[replication];
        for (size_t r = 0; r < replication; ++r) {
          rep_sum[r] = m_sum_init;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE_SUM_VAR_BODY(rep_sum[i%replication]);
        }

        for (size_t stride = replication/2; stride > 0; stride /= 2) {
          for (size_t r = 0; r < stride; ++r) {
            rep_sum[r] += rep_sum[r+stride];
          }
        }
        m_sum = rep_sum[0];

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  REDUCE_SUM : Unknown variant id = " << vid << std::endl;
    }

  }

}

void REDUCE_SUM::runSeqVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (tune_idx == t) {

    runSeqVariantDefault(vid);

  }

  t += 1;

  if ( vid == Base_Seq ) {

    seq_for(cpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        if (tune_idx == t) {

          runSeqVariantReplication<replication>(vid);

        }

        t += 1;

      }

    });

  }

}

void REDUCE_SUM::setSeqTuningDefinitions(VariantID vid)
{

  addVariantTuningName(vid, "default");

  if ( vid == Base_Seq ) {

    seq_for(cpu_atomic_replications_type{}, [&](auto replication) {

      if (run_params.numValidAtomicReplication() == 0u ||
          run_params.validAtomicReplication(replication)) {

        addVariantTuningName(vid, "replication_"+std::to_string(replication));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf
