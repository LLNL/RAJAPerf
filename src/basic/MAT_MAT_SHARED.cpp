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

namespace rajaperf {
namespace basic {

MAT_MAT_SHARED::MAT_MAT_SHARED(const RunParams &params)
    : KernelBase(rajaperf::Basic_MAT_MAT_SHARED, params) {
  setDefaultSize(1000);
  setDefaultReps(50);

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
  const Index_type NN = getRunSize() * getRunSize();
  allocAndInitData(m_A, NN, vid);
  allocAndInitData(m_B, NN, vid);
  allocAndInitData(m_C, NN, vid);
}

void MAT_MAT_SHARED::updateChecksum(VariantID vid) {
  checksum[vid] += calcChecksum(m_C, getRunSize());
}

void MAT_MAT_SHARED::tearDown(VariantID vid) {
  (void)vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
}

} // end namespace basic
} // end namespace rajaperf
