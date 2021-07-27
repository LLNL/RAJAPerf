//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_ni = 20; m_nj = 25; m_nk = 30;
      run_reps = 10000;
      break;
    case Small:
      m_ni = 60; m_nj = 70; m_nk = 80;
      run_reps = 1000;
      break;
    case Medium:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
    case Large:
      m_ni = 1000; m_nj = 1100; m_nk = 1200;
      run_reps = 1;
      break;
    case Extralarge:
      m_ni = 2000; m_nj = 2300; m_nk = 2600;
      run_reps = 1;
      break;
    default:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
  }

#if 0 // we want this...

  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1200;

  setDefaultProblemSize( ni_default * nj_default ) );
  setDefaultReps(4);

  m_ni = std::sqrt( getTargetProblemSize() ) + 1;
  m_nj = m_ni;
  m_nk = nk_default;
  
  m_alpha = 0.62;
  m_beta = 1.002;

#else // this is what we have now...

  m_alpha = 0.62;
  m_beta = 1.002;

  setDefaultProblemSize( m_ni * m_nj );
  setDefaultReps(run_reps);

#endif

  setActualProblemSize( m_ni * m_nj );

  setItsPerRep( m_ni * m_nj );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nk +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nk );
  setFLOPsPerRep((1 +
                  3 * m_nk) * m_ni*m_nj);

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
}

POLYBENCH_GEMM::~POLYBENCH_GEMM()
{
}

void POLYBENCH_GEMM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitDataConst(m_C, m_ni * m_nj, 0.0, vid);
}

void POLYBENCH_GEMM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_C, m_ni * m_nj);
}

void POLYBENCH_GEMM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
}

} // end namespace polybench
} // end namespace rajaperf
