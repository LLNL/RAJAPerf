//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_GEMM::POLYBENCH_GEMM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMM, params)
{
  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1200;

  setDefaultProblemSize( ni_default * nj_default );
  setDefaultReps(4);

  m_ni = std::sqrt( getTargetProblemSize() ) + std::sqrt(2)-1;
  m_nj = m_ni;
  m_nk = Index_type(double(nk_default)/ni_default*m_ni);

  m_alpha = 0.62;
  m_beta = 1.002;


  setActualProblemSize( m_ni * m_nj );

  setItsPerRep( m_ni * m_nj );
  setKernelsPerRep(1);
  setBytesReadPerRep( 1*sizeof(Real_type ) * m_ni * m_nk +
                      1*sizeof(Real_type ) * m_nj * m_nk );
  setBytesWrittenPerRep( 1*sizeof(Real_type ) * m_ni * m_nj);
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep((1 +
                  3 * m_nk) * m_ni*m_nj);

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setComplexity(Complexity::N_to_the_three_halves);

  setUsesFeature(Kernel);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_SYCL );
  setVariantDefined( RAJA_SYCL );
}

POLYBENCH_GEMM::~POLYBENCH_GEMM()
{
}

void POLYBENCH_GEMM::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitDataConst(m_C, m_ni * m_nj, 0.0, vid);
}

void POLYBENCH_GEMM::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_C, m_ni * m_nj, checksum_scale_factor , vid);
}

void POLYBENCH_GEMM::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_C, vid);
}

} // end namespace polybench
} // end namespace rajaperf
