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


void DAXPY::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;

  auto daxpy_lam = [=](Index_type i) {
                     DAXPY_BODY;
                   };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          DAXPY_BODY;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          daxpy_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), daxpy_lam);

      }
      stopTimer();

      break;
    }

    case RAJA_Vec : {

#if(0)
      DAXPY_DATA_VEC_SETUP;

      auto daxpy_vec_lam = [=](RAJA::VectorIndex<I, vector_t> i) {
                       DAXPY_VEC_BODY;
                    };
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forall<RAJA::vector_exec<vector_t>>(
          RAJA::TypedRangeSegment<I>(ibegin, iend), daxpy_vec_lam);
      }
      stopTimer();
#endif

#if(0)
      DAXPY_DATA_VEC_SETUP2;

      auto daxpy_vec_lam = [=](RAJA::VectorIndex<I, vector_t> i) {
                       DAXPY_VEC_BODY2;
                    };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        RAJA::forall<RAJA::vector_exec<vector_t>>(
          RAJA::TypedRangeSegment<I>(ibegin, iend), daxpy_vec_lam);
      }
      stopTimer();
#endif

#if(1)
      DAXPY_DATA_VEC_SETUP3;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        DAXPY_VEC_BODY3;
      }
      stopTimer();
#endif
      break;
    }
#endif //RUN_RAJA_VEC

    default : {
      std::cout << "\n  DAXPY : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
