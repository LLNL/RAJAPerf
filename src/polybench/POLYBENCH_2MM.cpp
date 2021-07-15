//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <algorithm>


namespace rajaperf
{
namespace polybench
{

POLYBENCH_2MM::POLYBENCH_2MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_2MM, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=22; m_nl=24;
      run_reps = 10000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=70; m_nl=80;
      run_reps = 1000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1100; m_nl=1200;
      run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2200; m_nl=2400;
      run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      run_reps = 100;
      break;
  }

#if 0 // we want this...

  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1120;
  Index_type nl_default = 1000;

  setDefaultProblemSize( std::max( ni_default*nj_default, 
                                   ni_default*nl_default ) );
  setDefaultReps(4);

  m_ni = std::sqrt( getTargetProblemSize() ) + 1;
  m_nj = m_ni;
  m_nk = nk_default;
  m_nl = m_ni;

  m_alpha = 1.5;
  m_beta = 1.2;

#else  // this is what we have now...

  m_alpha = 1.5;
  m_beta = 1.2;

  setDefaultProblemSize( std::max( m_ni*m_nj, m_ni*m_nl ) );
  setDefaultReps(run_reps);

#endif

  setActualProblemSize( std::max( m_ni*m_nj, m_ni*m_nl ) );

  setItsPerRep( m_ni*m_nj + m_ni*m_nl );
  setKernelsPerRep(2);
  setBytesPerRep( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nk +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nk +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nl );
  setFLOPsPerRep(3 * m_ni*m_nj*m_nk +
                 2 * m_ni*m_nj*m_nl );

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

POLYBENCH_2MM::~POLYBENCH_2MM()
{
}

void POLYBENCH_2MM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_tmp, m_ni * m_nj, vid);
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nl, vid);
  allocAndInitDataConst(m_D, m_ni * m_nl, 0.0, vid);
}

void POLYBENCH_2MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_D, m_ni * m_nl);
}

void POLYBENCH_2MM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_tmp);
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
}

} // end namespace polybench
} // end namespace rajaperf
