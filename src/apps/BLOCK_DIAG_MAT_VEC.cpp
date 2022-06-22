//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace apps {

BLOCK_DIAG_MAT_VEC::BLOCK_DIAG_MAT_VEC(const RunParams &params)
    : KernelBase(rajaperf::Apps_BLOCK_DIAG_MAT_VEC, params)
{
  m_N_default = 1000;
  setDefaultProblemSize(m_N_default);
  setDefaultReps(5);

  m_N = std::max(Index_type(getTargetProblemSize()), Index_type(1));

  setActualProblemSize(m_N);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( m_N*m_N*sizeof(Real_type) +
                  m_N*sizeof(Real_type) );

  //Square Mat-Vec product flops should be 2*N*N − N
  setFLOPsPerRep(2 * m_N * m_N - 1.0);

  checksum_scale_factor = 1e-6 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );



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

BLOCK_DIAG_MAT_VEC::~BLOCK_DIAG_MAT_VEC() {}

void BLOCK_DIAG_MAT_VEC::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type N = m_N;
  const Index_type ndof = m_ndof;
  const Index_type NE = Index_type(N/(ndof*ndof));

  allocAndInitDataConst(m_Me, ndof*ndof*NE, 0.0, vid);
  allocAndInitDataConst(m_X, ndof*NE, 0.0, vid);
  allocAndInitDataConst(m_Y, ndof*NE, 0.0, vid);
}

void BLOCK_DIAG_MAT_VEC::updateChecksum(VariantID vid, size_t tune_idx) {
  checksum[vid][tune_idx] += calcChecksum(m_Y, m_ndof*m_NE, checksum_scale_factor );
}

void BLOCK_DIAG_MAT_VEC::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  (void)vid;
  deallocData(m_Me);
  deallocData(m_X);
  deallocData(m_Y);
}

} // end namespace apps
} // end namespace rajaperf
