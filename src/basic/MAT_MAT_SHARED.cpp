//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
    : KernelBase(rajaperf::Basic_MAT_MAT_SHARED, params)
{
  m_N_default = 1000;
  setDefaultProblemSize(m_N_default*m_N_default);
  setDefaultReps(5);

  m_N = std::max(Index_type(std::sqrt(getTargetProblemSize())), Index_type(1));

  setActualProblemSize(m_N * m_N);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( m_N*m_N*sizeof(Real_type) +
                  m_N*m_N*sizeof(Real_type) );

  const Index_type no_tiles = (TL_SZ + m_N - 1) / TL_SZ;
  const Index_type no_blocks = RAJA_DIVIDE_CEILING_INT(m_N, TL_SZ);
  setFLOPsPerRep(2 * TL_SZ * TL_SZ * TL_SZ * no_tiles * no_blocks * no_blocks);

  checksum_scale_factor = 1e-6 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );


  setUsesFeature(Launch);

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

void MAT_MAT_SHARED::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type NN = m_N * m_N;

  allocAndInitDataConst(m_A, NN, 1.0, vid);
  allocAndInitDataConst(m_B, NN, 1.0, vid);
  allocAndInitDataConst(m_C, NN, 0.0, vid);
}

void MAT_MAT_SHARED::updateChecksum(VariantID vid, size_t tune_idx) {
  checksum[vid][tune_idx] += calcChecksum(m_C, m_N*m_N, checksum_scale_factor , vid);
}

void MAT_MAT_SHARED::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  (void)vid;
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_C, vid);
}

} // end namespace basic
} // end namespace rajaperf
