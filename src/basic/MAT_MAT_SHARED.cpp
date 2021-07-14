//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace basic {

MAT_MAT_SHARED::MAT_MAT_SHARED(const RunParams &params)
    : KernelBase(rajaperf::Basic_MAT_MAT_SHARED, params) {

  m_N_default = 1000;
  setDefaultProblemSize(m_N_default*m_N_default);
  setDefaultReps(50);

  m_N = std::max(Index_type(std::sqrt(getTargetProblemSize())), Index_type(1));

  setActualProblemSize(m_N * m_N);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( m_N*m_N*sizeof(Real_type) +
                  m_N*m_N*sizeof(Real_type) );

  const int no_tiles = (TL_SZ + m_N - 1) / TL_SZ;
  const int no_blocks = RAJA_DIVIDE_CEILING_INT(m_N, TL_SZ);
  setFLOPsPerRep(2 * TL_SZ * TL_SZ * TL_SZ * no_tiles * no_blocks * no_blocks);

  setUsesFeature(Teams);

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

MAT_MAT_SHARED::~MAT_MAT_SHARED() {}

void MAT_MAT_SHARED::setUp(VariantID vid) {
  const Index_type NN = m_N * m_N;
  allocAndInitData(m_A, NN, vid);
  allocAndInitData(m_B, NN, vid);
  allocAndInitData(m_C, NN, vid);
}

void MAT_MAT_SHARED::updateChecksum(VariantID vid) {
  checksum[vid] += calcChecksum(m_C, m_N*m_N);
}

void MAT_MAT_SHARED::tearDown(VariantID vid) {
  (void)vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
}

} // end namespace basic
} // end namespace rajaperf
