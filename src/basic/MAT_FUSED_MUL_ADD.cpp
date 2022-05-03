//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace basic {

MAT_FUSED_MUL_ADD::MAT_FUSED_MUL_ADD(const RunParams &params)
    : KernelBase(rajaperf::Basic_MAT_FUSED_MUL_ADD, params)
{
  m_N_default = 1024;
  setDefaultProblemSize(m_N_default);
  setDefaultReps(5);

  //Make sure problem target size is divisible by 16*16
  m_N = RAJA_DIVIDE_CEILING_INT(Index_type(getTargetProblemSize()),Index_type(m_Ne*m_Ne))*Index_type(m_Ne*m_Ne);
  setActualProblemSize(m_N);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep(2*m_N*sizeof(Real_type));
  setFLOPsPerRep(2*m_N*m_Ne);


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

MAT_FUSED_MUL_ADD::~MAT_FUSED_MUL_ADD() {}

void MAT_FUSED_MUL_ADD::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

  allocAndInitDataConst(m_A, m_N, 1.0, vid);
  allocAndInitDataConst(m_B, m_N, 1.0, vid);
  allocAndInitDataConst(m_D, m_N, 0.0, vid);

}

void MAT_FUSED_MUL_ADD::updateChecksum(VariantID vid, size_t tune_idx) {
  checksum[vid][tune_idx] += calcChecksum(m_D, m_N, checksum_scale_factor );
}

void MAT_FUSED_MUL_ADD::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  (void)vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_D);

}

} // end namespace basic
} // end namespace rajaperf
