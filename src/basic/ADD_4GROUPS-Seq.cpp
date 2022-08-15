//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void ADD_4GROUPS::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto a4g_lam = [=](Index_type i) {
        using mat_t = RAJA::expt::RectMatrixRegister<Real_type,RAJA::expt::RowMajorLayout,16, 4>;
        using row_t = RAJA::expt::RowIndex<int, mat_t>;
        using col_t = RAJA::expt::ColIndex<int, mat_t>;

        using Mat = RAJA::View<Real_type,RAJA::StaticLayout<RAJA::PERM_IJ,16,4>>;
        using Vec = RAJA::View<Real_type,RAJA::StaticLayout<RAJA::PERM_I,16>>;
        auto rall = row_t::static_all();
        auto call = col_t::static_all();

	Real_type * RAJA_RESTRICT  a = m_a;
	Real_type * RAJA_RESTRICT  b = m_b;
	Real_type * RAJA_RESTRICT  c = m_c;
        Real_type Y[16*4];
        auto aa   = Mat( a + 64 * i );
        auto bV   = Vec( b + 16 * i );
        auto cc   = Mat( c + 64 * i );
        auto yy   = Mat( Y );
	for(int j=0; j<16; j++){
            for(int k=0; k<4; k++){
                yy(j,k) = bV(j);
            }
        }
        cc(rall,call) = yy(rall,call) + aa(rall,call);
  };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
            Real_type* RAJA_RESTRICT  a = m_a + 64*i;
            Real_type* RAJA_RESTRICT  b = m_b + 16*i;
            Real_type* RAJA_RESTRICT  c = m_c + 64*i;
            for(Index_type j=0; j<16; j++){
                for(Index_type k=0; k<4; k++){
                    c[j*4+k] = a[j*4+k] + b[j];
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

        for (Index_type i = ibegin; i < iend; ++i ) {
          a4g_lam(i);
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), a4g_lam);

      }
      stopTimer();

      break;
    }
#endif

    default : {
      getCout() << "\n  ADD_4GROUPS : Unknown variant id = " << vid << std::endl;
    }

  }

}

} // end namespace basic
} // end namespace rajaperf
