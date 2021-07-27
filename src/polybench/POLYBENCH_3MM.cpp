//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <algorithm>


namespace rajaperf
{
namespace polybench
{


POLYBENCH_3MM::POLYBENCH_3MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_3MM, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=20; m_nl=22; m_nm=24;
      m_run_reps = 100000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=60; m_nl=70; m_nm=80;
      m_run_reps = 5000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1000; m_nl=1100; m_nm=1200;
      m_run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2000; m_nl=2200; m_nm=2400;
      m_run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
  }

#if 0 // we want this...

  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1010;
  Index_type nl_default = 1000;
  Index_type nm_default = 1200;

  setDefaultProblemSize( std::max( std::max( ni_default*nj_default, 
                                             nj_default*nl_default ), 
                                  ni_default*nl_default ) );
  setDefaultProblemSize( ni_default * nj_default );
  setDefaultReps(4);

  m_ni = std::sqrt( getTargetProblemSize() ) + 1;
  m_nj = m_ni;
  m_nk = nk_default;
  m_nl = m_ni;
  m_nm = nm_default;

#else  // this is what we have now...

  setDefaultProblemSize( std::max( std::max( m_ni*m_nj, m_nj*m_nl), m_ni*m_nl ) );

  setDefaultReps(m_run_reps);

#endif

  setActualProblemSize( std::max( std::max( m_ni*m_nj, m_nj*m_nl ), 
                                  m_ni*m_nl ) );

  setItsPerRep( m_ni*m_nj + m_nj*m_nl + m_ni*m_nl );
  setKernelsPerRep(3);
  setBytesPerRep( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nk +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nk +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_nj * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nm +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nl * m_nm +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nl );
  setFLOPsPerRep(2 * m_ni*m_nj*m_nk +
                 2 * m_nj*m_nl*m_nm +
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

POLYBENCH_3MM::~POLYBENCH_3MM()
{
}

void POLYBENCH_3MM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nm, vid);
  allocAndInitData(m_D, m_nm * m_nl, vid);
  allocAndInitDataConst(m_E, m_ni * m_nj, 0.0, vid);
  allocAndInitDataConst(m_F, m_nj * m_nl, 0.0, vid);
  allocAndInitDataConst(m_G, m_ni * m_nl, 0.0, vid);
}

void POLYBENCH_3MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_G, m_ni * m_nl);
}

void POLYBENCH_3MM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
  deallocData(m_E);
  deallocData(m_F);
  deallocData(m_G);
}

} // end namespace basic
} // end namespace rajaperf
