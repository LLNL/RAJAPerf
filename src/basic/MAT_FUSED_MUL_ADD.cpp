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
  m_N_default = 16;
  setDefaultProblemSize(m_N_default);
  setDefaultReps(5);

  //If problem target size is not divisible by Ne, round up
//  m_N = std::max(Index_type(getTargetProblemSize())*(Index_type(getTargetProblemSize())/16), \
//  		Index_type(m_Ne));

  m_N = 16;
  setActualProblemSize(m_N*m_N);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( m_N*m_N*sizeof(Real_type) +
                  m_N*m_N*sizeof(Real_type) );

  //Square Mat-Mat product flops should be (2^N−1)N^2=2*N^3−N^2
  setFLOPsPerRep(2*m_N*m_N*m_N - m_N*m_N);

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
	
  //hard coded for 16 at the moment
  const Index_type m_Me = m_Ne;
  const Index_type m_Ne = m_Ne;
  const Index_type m_Ke = m_Ne;
  //global matrices
  allocAndInitDataConst(m_A, m_N * m_N, 1.0, vid);
  allocAndInitDataConst(m_B, m_N * m_N, 1.0, vid);
  allocAndInitDataConst(m_D, m_N * m_N, 0.0, vid);
  //element/batch matrices
  allocAndInitDataConst(m_Ae, m_Me * m_Ke, 1.0, vid);
  allocAndInitDataConst(m_Be, m_Ke * m_Ne, 1.0, vid);
  allocAndInitDataConst(m_De, m_Me * m_Ne, 0.0, vid);
}

void MAT_FUSED_MUL_ADD::updateChecksum(VariantID vid, size_t tune_idx) {
  checksum[vid][tune_idx] += calcChecksum(m_Ae, m_N, checksum_scale_factor );
}

void MAT_FUSED_MUL_ADD::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  (void)vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_D);
  deallocData(m_Ae);
  deallocData(m_Be);
  deallocData(m_De);
}

} // end namespace basic
} // end namespace rajaperf
