//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL_PAR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void TRIDIAGONAL_PAR::runSeqVariant(VariantID vid, size_t /*tune_idx*/)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_PAR_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      // TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_V2;

          Real_type Aa_[N-1]; // lower diagonal of A
          Real_ptr Aa = Aa_ - 2; // [2:N]
          for (int n = 2; n <= N; ++n) { // par
            Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);
            Aa[n] = Aa_data[idx_m];
          }
          Real_type Ac_[N-1]; // upper diagonal of A
          Real_ptr Ac = Ac_ - 1; // [1:N-1]
          for (int n = 1; n <= N-1; ++n) { // par
            Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);
            Ac[n] = Ac_data[idx_m];
          }
          Real_type Ab_[N]; // diagonal of A
          Real_ptr Ab = Ab_ - 1; // [1:N]
          for (int n = 1; n <= N; ++n) { // par
            Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);
            Ab[n] = Ab_data[idx_m];
          }
          Real_type b_[N]; // rhs of equation
          Real_ptr b = b_ - 1; // [1:N]
          for (int n = 1; n <= N; ++n) { // par
            Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);
            b[n] = b_data[idx_m];
          }

          Real_type AaAc_[N]; // holds products (-e[i]*f[i-1])
          Real_ptr AaAc = AaAc_ - 1; // [1:N]
          Real_type temp_[N]; // temporary array
          Real_ptr temp = temp_ - 1; // [1:N]
          Real_type qi_[N]; // Qi[n]
          Real_ptr qi = qi_ - 1; // [1:N]
          Real_type qim1_[N+1]; // Qi-1[n]
          Real_ptr qim1 = qim1_ - 0; // [0:N]
          Real_type qim2_[N+2]; // Qi-1[n]
          Real_ptr qim2 = qim2_ - (-1); // [-1:N]

          Real_type u_[N];
          Real_ptr u = u_ - 1; // [1:N]

          // Real_type m_[N-1];
          // Real_ptr m = m_ - 2; // [2:N]
          Real_type m_[N];
          Real_ptr m = m_ - 1; // [1:N]

          Real_type x_[N];
          Real_ptr x = x_ - 1; // [1:N]

          AaAc[1] = 0;
          for (int n = 2; n <= N; ++n) { // par
            AaAc[n] = -Aa[n] * Ac[n-1];
          }
          for (int n = -1; n <= N; ++n) { // par
            qim2[n] = 1;
          }
          qim1[0] = 1;
          for (int n = 1; n <= N; ++n) { // par
            qim1[n] = Ab[n];
          }
          qi[1] = Ab[1];
          for (int n = 2; n <= N; ++n) { // par
            qi[n] = Ab[n] * Ab[n-1] + AaAc[n];
          }
          for (int k = 2; k <= N; k *= 2) {
            for (int n = k-1; n <= N; ++n) { // par
              temp[n] = qim1[n] * qim1[n-k+1] + AaAc[n-k+2] * qim2[n] * qim2[n-k];
            }
            for (int n = N; n >= k; --n) { // par (beware)
              qim1[n] = qi[n] * qim1[n-k] + AaAc[n-k+1] * qim1[n] * qim2[n-k-1];
            }
            for (int n = k-1; n <= N; ++n) { // par
              qim2[n] = temp[n];
            }
            for (int n = k+1; n <= N; ++n) { // par
              qi[n] = Ab[n] * qim1[n-1] + AaAc[n] * qim2[n-2];
            }
          }

          u[1] = qi[1];
          for (int n = 2; n <= N; ++n) { // par
            u[n] = qi[n] / qi[n-1];
          }
          for (int n = 2; n <= N; ++n) { // par
            m[n] = Aa[n] / u[n-1];
          }
          for (int n = 1; n <= N; ++n) { // par
            x[n] = b[n];
          }
          m[1] = 0;
          for (int n = 2; n <= N; ++n) { // par
            m[n] = -m[n];
          }

          for (int k = 1; k <= N; k *= 2) {
            for (int n = N; n >= k+1; --n) { // par (beware)
              x[n] = x[n] + x[n-k] * m[n];
            }
            for (int n = N; n >= k+1; --n) { // par (beware)
              m[n] = m[n] * m[n-k];
            }
          }

          for (int n = 1; n <= N; ++n) { // par
            x[n] = x[n] / u[n];
          }
          for (int n = 1; n <= N-1; ++n) { // par
            m[n] = -Ac[n] / u[n];
          }
          m[N] = 0;
          for (int k = 1; k <= N; k *= 2) {
            for (int n = 1; n <= N-k; ++n) { // par (beware)
              x[n] = x[n] + x[n+k] * m[n];
            }
            for (int n = 1; n <= N-k; ++n) { // par (beware)
              m[n] = m[n] * m[n+k];
            }
          }

          for (int n = 1; n <= N; ++n) { // par
            Index_type idx_m = TRIDIAGONAL_PAR_INDEX(n-1);
            x_data[idx_m] = x[n];
          }

        }

      }
      stopTimer();

      // TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL;
                         TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          triad_lam(i);
        }

      }
      stopTimer();

      TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }

    case RAJA_Seq : {

      TRIDIAGONAL_PAR_TEMP_DATA_SETUP_LOCAL;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_LOCAL;
                         TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_LOCAL;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), triad_lam);

      }
      stopTimer();

      TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_LOCAL;

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  TRIDIAGONAL_PAR : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace stream
} // end namespace rajaperf
