//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SW4CK_KERNEL_5.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void SW4CK_KERNEL_5::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  //To be populated later with
  const int istart = 0;
  const int ifirst = 0;
  const int ilast = 1;
  const int jstart = 0;
  const int jfirst = 0;
  const int jend = 0;
  const int jlast = 1;
  const int kfirst = 0;
  const int kstart = 0;
  const int klast = 1;
  const int kend = 1;

  const int nk = 0;
  
  char op = '=';

  SW4CK_KERNEL_5_DATA_SETUP;


  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //Reference impl
        for(int k=kstart; k<kend+1; k++) {
          for(int j=jstart; j<jend+1; j++) {
            for(int i=istart; i<ilast - 1; i++) {

              // 5 ops
              SW4CK_KERNEL_5_BODY_1;

              // pp derivative (u) (u-eq)
              // 53 ops, tot=58
              SW4CK_KERNEL_5_BODY_2;

              // qq derivative (u) (u-eq)
              // 43 ops, tot=101
              SW4CK_KERNEL_5_BODY_3;

              // pp derivative (v) (v-eq)
              // 43 ops, tot=144
              SW4CK_KERNEL_5_BODY_4;

              // qq derivative (v) (v-eq)
              // 53 ops, tot=197
              SW4CK_KERNEL_5_BODY_4;

              // qq derivative (v) (v-eq)
              // 53 ops, tot=197
              SW4CK_KERNEL_5_BODY_5;

              // pp derivative (w) (w-eq)
              // 43 ops, tot=240
              SW4CK_KERNEL_5_BODY_6;

              // qq derivative (w) (w-eq)
              // 43 ops, tot=283
              SW4CK_KERNEL_5_BODY_7;

              // All rr-derivatives at once
              // averaging the coefficient
              // 54*8*8+25*8 = 3656 ops, tot=3939
              SW4CK_KERNEL_5_BODY_8;

              //Magic sync goes here for the GPU versions              
              SW4CK_KERNEL_5_BODY_8_1;

              //AMD UNROLL FIX goes here
              SW4CK_KERNEL_5_BODY_8_2;

              // Ghost point values, only nonzero for k=nk.
              // 72 ops., tot=4011
              SW4CK_KERNEL_5_BODY_9;

              // pq-derivatives (u-eq)
              // 38 ops., tot=4049
              SW4CK_KERNEL_5_BODY_10;

              // qp-derivatives (u-eq)
              // 38 ops. tot=4087
              SW4CK_KERNEL_5_BODY_11;

              // pq-derivatives (v-eq)
              // 38 ops. , tot=4125
              SW4CK_KERNEL_5_BODY_12;

              // qp-derivatives (v-eq)
              // 38 ops., tot=4163
              SW4CK_KERNEL_5_BODY_13;

              // rp - derivatives
              // 24*8 = 192 ops, tot=4355
              SW4CK_KERNEL_5_BODY_14;

              // rp derivatives (u-eq)
              // 67 ops, tot=4422
              SW4CK_KERNEL_5_BODY_15;

              // rp derivatives (v-eq)
              // 42 ops, tot=4464
              SW4CK_KERNEL_5_BODY_16;

              // rp derivatives (w-eq)
              // 38 ops, tot=4502
              SW4CK_KERNEL_5_BODY_17;

              // rq - derivatives
              // 24*8 = 192 ops , tot=4694
              SW4CK_KERNEL_5_BODY_18;

              // rq derivatives (u-eq)
              // 42 ops, tot=4736
              SW4CK_KERNEL_5_BODY_19;

              // rq derivatives (v-eq)
              // 70 ops, tot=4806
              SW4CK_KERNEL_5_BODY_20;

              // rq derivatives (w-eq)
              // 39 ops, tot=4845
              SW4CK_KERNEL_5_BODY_21;

              // pr and qr derivatives at once
              // in loop: 8*(53+53+43) = 1192 ops, tot=6037              
              SW4CK_KERNEL_5_BODY_22;

              // 12 ops, tot=6049          
              SW4CK_KERNEL_5_BODY_23;
            }
          }
        }



      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //Lambda impl

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //RAJA impl

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  SW4CK_KERNEL_5 : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace apps
} // end namespace rajaperf
