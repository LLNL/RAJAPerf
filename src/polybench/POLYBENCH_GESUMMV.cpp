//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_GESUMMV::POLYBENCH_GESUMMV(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GESUMMV, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_N = 30;
      run_reps = 10000;
      break;
    case Small:
      m_N = 90;
      run_reps = 1000;
      break;
    case Medium:
      m_N = 250;
      run_reps = 100;
      break;
    case Large:
      m_N = 1300;
      run_reps = 1;
      break;
    case Extralarge:
      m_N = 2800;
      run_reps = 1;
      break;
    default:
      m_N = 1600;
      run_reps = 120;
      break;
  }

#if 0 // we want this...

  Index_type N_default = 1000;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(120); 

  m_N = std::sqrt( getTargetProblemSize() ) + 1;

  m_alpha = 0.62;
  m_beta = 1.002;

#else // this is what we have now...

  m_alpha = 0.62;
  m_beta = 1.002;

  setDefaultProblemSize( m_N * m_N );
  setDefaultReps(run_reps);

#endif

  setActualProblemSize( m_N * m_N );

  setItsPerRep( m_N );
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N +
                  (0*sizeof(Real_type ) + 2*sizeof(Real_type )) * m_N * m_N );
  setFLOPsPerRep((4 * m_N +
                  3 ) * m_N  );

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
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

POLYBENCH_GESUMMV::~POLYBENCH_GESUMMV()
{
}

void POLYBENCH_GESUMMV::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_x, m_N, vid);
  allocAndInitDataConst(m_y, m_N, 0.0, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitData(m_B, m_N * m_N, vid);
}

void POLYBENCH_GESUMMV::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, m_N);
}

void POLYBENCH_GESUMMV::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_A);
  deallocData(m_B);
}

} // end namespace polybench
} // end namespace rajaperf
