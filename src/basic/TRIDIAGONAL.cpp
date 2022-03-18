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

  checksum_scale_factor = (1e0 / m_N) *
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

  allocAndInitDataRandValue(m_Aa_global, size, vid);
  allocAndInitDataRandValue(m_Ab_global, size, vid);
  allocAndInitDataRandValue(m_Ac_global, size, vid);
  allocAndInitDataConst(m_x_global, size, 0.0, vid);
  allocAndInitDataRandValue(m_b_global, size, vid);
}

void TRIDIAGONAL::updateChecksum(VariantID vid, size_t tune_idx) {
  checksum[vid][tune_idx] += calcChecksum(m_x_global, m_N * getActualProblemSize(), checksum_scale_factor );
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
