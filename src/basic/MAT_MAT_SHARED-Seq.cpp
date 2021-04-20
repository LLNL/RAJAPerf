//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void MAT_MAT_SHARED::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  MAT_MAT_SHARED_DATA_SETUP;

  /*
  auto mas_lam = [=](Index_type i) {
    //MAT_MAT_SHARED_BODY;
                 };
  */
  const int N = 1000;
  const int Nx = (N-1)/TL_SZ+1;
  const int Ny = (N-1)/TL_SZ+1;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //Write Sequential variant here
        for(int by = 0; by < Ny; ++by){
          for(int bx = 0; bx < Nx; ++bx){
            
            MAT_MAT_SHARED_BODY_0
            
            for(int ty=0; ty<TL_SZ; ++ty){
              for(int tx=0; tx<TL_SZ; ++tx){
                MAT_MAT_SHARED_BODY_1
              }
            }
            
            //Sequential loop
            for(k = 0; k < (TL_SZ + N - 1)/TL_SZ; ++k) {
              
              for(int ty=0; ty<TL_SZ; ++ty){
                for(int tx=0; tx<TL_SZ; ++tx){
                  
                  MAT_MAT_SHARED_BODY_2
                  
                }
              }
              
              //synchronize();
              for(int ty=0; ty<TL_SZ; ++ty){
                for(int tx=0; tx<TL_SZ; ++tx){
                  
                  MAT_MAT_SHARED_BODY_3

                }
              }
              
            }//Sequential loop
            
            for(int ty=0; ty<TL_SZ; ++ty){
              for(int tx=0; tx<TL_SZ; ++tx){
                MAT_MAT_SHARED_BODY_4
              }
            }
            
          }
        }                

      }//number of iterations
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (Index_type irep = 0; irep < run_reps; ++irep) {

        //for (Index_type i = ibegin; i < iend; ++i ) {
        //mas_lam(i);
        //}

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA::forall<RAJA::simd_exec>(
        //RAJA::RangeSegment(ibegin, iend), mas_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
