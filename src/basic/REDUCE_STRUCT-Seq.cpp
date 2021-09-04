//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf 
{
namespace basic
{


void REDUCE_STRUCT::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

  	  for (int i=0;i<particles.N+1;i++){
  	      particles.x[i] = i*dx;  
  	      particles.y[i] = i*dy; 
	  }

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type xsum = 0.0;
      Real_type xmin = 0.0;
      Real_type xmax = 0.0;

      for (Index_type i = ibegin; i < iend; ++i ) {
        REDUCE_STRUCT_BODY;
      }

      particles.SetCenter(xsum/particles.N,0.0);
      particles.SetXMin(xmin);
      particles.SetXMax(xmax);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      for (int i=0;i<particles.N+1;i++){
          particles.x[i] = i*dx;  
          particles.y[i] = i*dy; 
      } 

      auto init_struct_base_lam = [=](Index_type i) -> Real_type {
                              return particles.x[i];
                            };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type xsum = 0.0;
      Real_type xmin = 0.0;
      Real_type xmax = 0.0;

      for (Index_type i = ibegin; i < iend; ++i ) {
        xsum += init_struct_base_lam(i);
        xmin = RAJA_MIN(xmin, init_struct_base_lam(i));
        xmax = RAJA_MAX(xmax, init_struct_base_lam(i));
      }
      particles.SetCenter(xsum/particles.N,0.0);
      particles.SetXMin(xmin);
      particles.SetXMax(xmax);

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      //startTimer();
      //for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //  RAJA::ReduceSum<RAJA::seq_reduce, Int_type> vsum(m_vsum_init);
      //  RAJA::ReduceMin<RAJA::seq_reduce, Int_type> vmin(m_vmin_init);
      //  RAJA::ReduceMax<RAJA::seq_reduce, Int_type> vmax(m_vmax_init);

      //  RAJA::forall<RAJA::loop_exec>(
      //    RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
      //    REDUCE3_INT_BODY_RAJA;
      //  });

      //  m_vsum += static_cast<Int_type>(vsum.get());
      //  m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      //  m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

      //}
      //stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  REDUCE_STRUCT : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
