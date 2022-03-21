//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace basic {

TRIDIAGONAL::TRIDIAGONAL(const RunParams &params)
    : KernelBase(rajaperf::Basic_TRIDIAGONAL, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(10);

  m_N = 73;

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);
  setBytesPerRep( (m_N*sizeof(Real_type) + 4*m_N*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep((2 + 6 * (m_N-1) +
                  0 + 2 * (m_N-1)
                  ) * getActualProblemSize());

  checksum_scale_factor = (1e-3 / m_N) *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );


  setUsesFeature(Forall);

  setVariantDefined(Base_Seq);
  setVariantDefined(Lambda_Seq);
  setVariantDefined(RAJA_Seq);

  setVariantDefined(Base_OpenMP);
  setVariantDefined(Lambda_OpenMP);
  setVariantDefined(RAJA_OpenMP);

  setVariantDefined(Base_CUDA);
  setVariantDefined(Lambda_CUDA);
  setVariantDefined(RAJA_CUDA);

  setVariantDefined(Base_HIP);
  setVariantDefined(Lambda_HIP);
  setVariantDefined(RAJA_HIP);
}

TRIDIAGONAL::~TRIDIAGONAL() {}

void TRIDIAGONAL::setUp(VariantID vid, size_t /*tune_idx*/) {
  const Index_type size = m_N * getActualProblemSize();
  // Using custom seeds so the matrix has a solution
  allocAndInitDataRandValue(m_Aa_global, size, vid, 123456);
  allocAndInitDataRandValue(m_Ab_global, size, vid, 234567);
  allocAndInitDataRandValue(m_Ac_global, size, vid, 345678);
  allocAndInitDataConst(m_x_global, size, 0.0, vid);
  allocAndInitDataRandValue(m_b_global, size, vid, 456789);
}

void TRIDIAGONAL::updateChecksum(VariantID vid, size_t tune_idx) {
  // calculate checksum as sum of errors, instead of sum of x
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();
  TRIDIAGONAL_DATA_SETUP;
  for (Index_type i = ibegin; i < iend; ++i) {
    TRIDIAGONAL_LOCAL_DATA_SETUP;
    Index_type idx_0 = TRIDIAGONAL_INDEX(0);
    Index_type idx_1 = TRIDIAGONAL_INDEX(1);
    Real_type b0 = Ab[idx_0] * x[idx_0] +
                   Ac[idx_0] * x[idx_1];
    checksum[vid][tune_idx] += std::abs(b0 - b[idx_0]) * checksum_scale_factor;
    for (Index_type n = 1; n < N-1; ++n) {
      Index_type idx_m = TRIDIAGONAL_INDEX(n-1);
      Index_type idx_n = TRIDIAGONAL_INDEX(n);
      Index_type idx_p = TRIDIAGONAL_INDEX(n+1);
      Real_type bn = Aa[idx_n] * x[idx_m] +
                     Ab[idx_n] * x[idx_n] +
                     Ac[idx_n] * x[idx_p];
      checksum[vid][tune_idx] += std::abs(bn - b[idx_n]) * checksum_scale_factor;
    }
    Index_type idx_M = TRIDIAGONAL_INDEX(N-2);
    Index_type idx_N = TRIDIAGONAL_INDEX(N-1);
    Real_type bN = Aa[idx_N] * x[idx_M] +
                   Ab[idx_N] * x[idx_N];
    checksum[vid][tune_idx] += std::abs(bN - b[idx_N]) * checksum_scale_factor;
  }
}

void TRIDIAGONAL::tearDown(VariantID vid, size_t /*tune_idx*/) {
  (void)vid;
  deallocData(m_Aa_global);
  deallocData(m_Ab_global);
  deallocData(m_Ac_global);
  deallocData(m_x_global);
  deallocData(m_b_global);
}

} // end namespace basic
} // end namespace rajaperf
