//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace polybench
{


POLYBENCH_ADI::POLYBENCH_ADI(const RunParams& params)
  : KernelBase(rajaperf::Polybench_ADI, params)
{
#if 0
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps;
  switch(lsizespec) {
    case Mini:
      m_n=20; m_tsteps=1;
      run_reps = 10000;
      break;
    case Small:
      m_n=60; m_tsteps=40;
      run_reps = 500;
      break;
    case Medium:
      m_n=200; m_tsteps=100;
      run_reps = 20;
      break;
    case Large:
      m_n=1000; m_tsteps=500;
      run_reps = 1;
      break;
    case Extralarge:
      m_n=2000; m_tsteps=1000;
      run_reps = 1;
      break;
    default:
      m_n=200; m_tsteps=100;
      run_reps = 20;
      break;
  }

  setDefaultProblemSize( (m_n-2)*(m_n-2) );
  setDefaultReps(run_reps);

  setItsPerRep( m_tsteps * ( (m_n-2)*(m_n-2 + m_n-2) +
                             (m_n-2)*(m_n-2 + m_n-2) ) );

#else

  Index_type n_default = 1000;
  
  setDefaultProblemSize( (n_default-2) * (n_default-2) );
  setDefaultReps(4);

  m_n = std::sqrt( getTargetProblemSize() ) + 1;
  m_tsteps = 4;

  setItsPerRep( m_tsteps * ( (m_n-2) + (m_n-2) ) );

#endif

  setActualProblemSize( (m_n-2) * (m_n-2) );

  setKernelsPerRep( m_tsteps * 2 );
  setBytesPerRep( m_tsteps * ( (3*sizeof(Real_type ) + 3*sizeof(Real_type )) * m_n * (m_n-2) +
                               (3*sizeof(Real_type ) + 3*sizeof(Real_type )) * m_n * (m_n-2) ) );
  setFLOPsPerRep( m_tsteps * ( (15 + 2) * (m_n-2)*(m_n-2) +
                               (15 + 2) * (m_n-2)*(m_n-2) ) );

  checksum_scale_factor = 0.0000001 * 
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

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

POLYBENCH_ADI::~POLYBENCH_ADI()
{
}

void POLYBENCH_ADI::setUp(VariantID vid)
{
  allocAndInitDataConst(m_U, m_n * m_n, 0.0, vid);
  allocAndInitData(m_V, m_n * m_n, vid);
  allocAndInitData(m_P, m_n * m_n, vid);
  allocAndInitData(m_Q, m_n * m_n, vid);
}

void POLYBENCH_ADI::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_U, m_n * m_n, checksum_scale_factor );
}

void POLYBENCH_ADI::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_U);
  deallocData(m_V);
  deallocData(m_P);
  deallocData(m_Q);
}

} // end namespace polybench
} // end namespace rajaperf
