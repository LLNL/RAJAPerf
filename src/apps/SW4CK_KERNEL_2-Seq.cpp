//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SW4CK_KERNEL_2.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void SW4CK_KERNEL_2::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  SW4CK_KERNEL_2_DATA_SETUP;

  //To be populated later with
  const int ifirst = 0;
  const int ilast = 1;
  const int jfirst = 0;
  const int jlast = 1;
  const int kstart = 0;
  const int klast = 1;
  const int kend = 1;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        //Reference impl
        for(int k=kstart; k<kend+1; k++) {
          for(int j=jstart; j<jend+1; j++) {
            for(int i=istart; i<ilast - 1; i++) {

              // 5 ops
              SW4CK_KERNEL_2_BODY_1;

              // pp derivative (u)
              // 53 ops, tot=58
              SW4CK_KERNEL_2_BODY_2;

              // qq derivative (u)
              // 43 ops, tot=101
              SW4CK_KERNEL_2_BODY_3;

              // rr derivative (u)
              // 5*11+14+14=83 ops, tot=184
              SW4CK_KERNEL_2_BODY_4;

              // rr derivative (v)
              // 42 ops, tot=226
              SW4CK_KERNEL_2_BODY_5;

              // rr derivative (w)
              // 43 ops, tot=269
              SW4CK_KERNEL_2_BODY_6;

              // pq-derivatives
              // 38 ops, tot=307
              SW4CK_KERNEL_2_BODY_7;

              // qp-derivatives
              // 38 ops, tot=345
              SW4CK_KERNEL_2_BODY_8;

              // pr-derivatives
              // 130 ops., tot=475
              SW4CK_KERNEL_2_BODY_9;

              // rp derivatives
              // 130 ops, tot=605
              SW4CK_KERNEL_2_BODY_10;

              // qr derivatives
              // 82 ops, tot=687
              SW4CK_KERNEL_2_BODY_11;

              // rq derivatives
              // 82 ops, tot=769
              SW4CK_KERNEL_2_BODY_12;

              // 4 ops, tot=773
              SW4CK_KERNEL_2_BODY_13;

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
      getCout() << "\n  SW4CK_KERNEL_2 : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace apps
} // end namespace rajaperf
