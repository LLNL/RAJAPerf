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

      TRIDIAGONAL_PAR_TEMP_DATA_SETUP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;

          double E_[N-1]; // lower diagonal of A
          double* E = E_ - 2; // [2:N]
          for (int j = 2; j <= N; ++j) { // par
            E[j] = Aa[j-1];
          }
          double F_[N-1]; // upper diagonal of A
          double* F = F_ - 1; // [1:N-1]
          for (int j = 1; j <= N-1; ++j) { // par
            F[j] = Ac[j-1];
          }
          double D_[N]; // diagonal of A
          double* D = D_ - 1; // [1:N]
          for (int j = 1; j <= N; ++j) { // par
            D[j] = Ab[j-1];
          }
          double B_[N]; // rhs of equation
          double* B = B_ - 1; // [1:N]
          for (int j = 1; j <= N; ++j) { // par
            B[j] = b[j-1];
          }

          double EF_[N]; // holds products (-e[i]*f[i-1])
          double* EF = EF_ - 1; // [1:N]
          double TEMP_[N]; // temporary array
          double* TEMP = TEMP_ - 1; // [1:N]
          double QI_[N]; // Qi[j]
          double* QI = QI_ - 1; // [1:N]
          double QIM1_[N+1]; // Qi-1[j]
          double* QIM1 = QIM1_ - 0; // [0:N]
          double QIM2_[N+2]; // Qi-1[j]
          double* QIM2 = QIM2_ - (-1); // [-1:N]

          double U_[N];
          double* U = U_ - 1; // [1:N]

          // double M_[N-1];
          // double* M = M_ - 2; // [2:N]
          double M_[N];
          double* M = M_ - 1; // [1:N]

          double Y_[N];
          double* Y = Y_ - 1; // [1:N]

          double X_[N];
          double* X = X_ - 1; // [1:N]

          EF[1] = 0;
          for (int j = 2; j <= N; ++j) { // par
            EF[j] = -E[j] * F[j-1];
          }
          for (int j = -1; j <= N; ++j) { // par
            QIM2[j] = 1;
          }
          QIM1[0] = 1;
          for (int j = 1; j <= N; ++j) { // par
            QIM1[j] = D[j];
          }
          QI[1] = D[1];
          for (int j = 2; j <= N; ++j) { // par
            QI[j] = D[j] * D[j-1] + EF[j];
          }
          for (int i = 2; i <= N; i *= 2) {
            for (int j = i-1; j <= N; ++j) { // par
              TEMP[j] = QIM1[j] * QIM1[j-i+1] + EF[j-i+2] * QIM2[j] * QIM2[j-i];
            }
            for (int j = N; j >= i; --j) { // par (beware)
              QIM1[j] = QI[j] * QIM1[j-i] + EF[j-i+1] * QIM1[j] * QIM2[j-i-1];
            }
            for (int j = i-1; j <= N; ++j) { // par
              QIM2[j] = TEMP[j];
            }
            for (int j = i+1; j <= N; ++j) { // par
              QI[j] = D[j] * QIM1[j-1] + EF[j] * QIM2[j-2];
            }
          }

          U[1] = QI[1];
          for (int j = 2; j <= N; ++j) { // par
            U[j] = QI[j] / QI[j-1];
          }
          for (int j = 2; j <= N; ++j) { // par
            M[j] = E[j] / U[j-1];
          }
          for (int j = 1; j <= N; ++j) { // par
            Y[j] = B[j];
          }
          M[1] = 0;
          for (int j = 2; j <= N; ++j) { // par
            M[j] = -M[j];
          }

          for (int i = 1; i <= N; i *= 2) {
            for (int j = N; j >= i+1; --j) { // par (beware)
              Y[j] = Y[j] + Y[j-i] * M[j];
            }
            for (int j = N; j >= i+1; --j) { // par (beware)
              M[j] = M[j] * M[j-i];
            }
          }

          for (int j = 1; j <= N; ++j) { // par
            X[j] = Y[j] / U[j];
          }
          for (int j = 1; j <= N-1; ++j) { // par
            M[j] = -F[j] / U[j];
          }
          M[N] = 0;
          for (int i = 1; i <= N; i *= 2) {
            for (int j = 1; j <= N-i; ++j) { // par (beware)
              X[j] = X[j] + X[j+i] * M[j];
            }
            for (int j = 1; j <= N-i; ++j) { // par (beware)
              M[j] = M[j] * M[j+i];
            }
          }

          for (int i = 1; i <= N; ++i) { // par
            x[i-1] = X[i];
          }

        }

      }
      stopTimer();

      TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN;

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      TRIDIAGONAL_PAR_TEMP_DATA_SETUP;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_PAR_BODY_FORWARD;
                         TRIDIAGONAL_PAR_BODY_BACKWARD;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          triad_lam(i);
        }

      }
      stopTimer();

      TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN;

      break;
    }

    case RAJA_Seq : {

      TRIDIAGONAL_PAR_TEMP_DATA_SETUP;

      auto triad_lam = [=](Index_type i) {
                         TRIDIAGONAL_PAR_LOCAL_DATA_SETUP;
                         TRIDIAGONAL_PAR_BODY_FORWARD;
                         TRIDIAGONAL_PAR_BODY_BACKWARD;
                       };

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), triad_lam);

      }
      stopTimer();

      TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN;

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
