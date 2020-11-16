//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

 
void ADD::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ADD_DATA_SETUP;

#ifdef RUN_RAJA_VEC
  using vector_t = RAJA::StreamVector<double,2>;
  using index_t = ptrdiff_t;

  RAJA::View<double, RAJA::Layout<1>> A(a, iend);
  RAJA::View<double, RAJA::Layout<1>> B(b, iend);
  RAJA::View<double, RAJA::Layout<1>> C(c, iend);

  auto add_lam = [=](RAJA::VectorIndex<index_t, vector_t> i) {
                   C(i) = A(i) + B(i);
                 };
#else

  auto add_lam = [=](Index_type i) {
                   ADD_BODY;
                 };

#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
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
          add_lam(i);
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), add_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RUN_RAJA_VEC)
    case RAJA_Vec : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        RAJA::forall<RAJA::vector_exec<vector_t>>(
          RAJA::TypedRangeSegment<index_t>(ibegin, iend), add_lam);

      }
      stopTimer();

      break;
    }
    
#endif //RUN_RAJA_VEC

    default : {
      std::cout << "\n  ADD : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace stream
} // end namespace rajaperf
