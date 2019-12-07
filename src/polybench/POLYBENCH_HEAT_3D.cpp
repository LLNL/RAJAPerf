//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

 
POLYBENCH_HEAT_3D::POLYBENCH_HEAT_3D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_HEAT_3D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
//
// Note: 'factor' was added to keep the checksums (which can get very large
//       for this kernel) within a reasonable range for comparison across
//       variants.
//
  switch(lsizespec) {
    case Mini:
      m_N=10;
      m_tsteps=20;
      run_reps = 1000;
      m_factor = 0.1;
      break;
    case Small:
      m_N=20;
      m_tsteps=40;
      run_reps = 500;
      m_factor = 0.01;
      break;
    case Medium:
      m_N=40;
      m_tsteps=100;
      run_reps = 300;
      m_factor = 0.001;
      break;
    case Large:
      m_N=120;
      m_tsteps=50;
      run_reps = 10;
      m_factor = 0.0001;
      break;
    case Extralarge:
      m_N=400;
      m_tsteps=10;
      run_reps = 1;
      m_factor = 0.00001;
      break;
    default:
      m_N=120;
      m_tsteps=20;
      run_reps = 10;
      m_factor = 0.0001;
      break;
  }

  setDefaultSize( m_tsteps * 2 * m_N * m_N * m_N);
  setDefaultReps(run_reps);
}

POLYBENCH_HEAT_3D::~POLYBENCH_HEAT_3D() 
{
}

void POLYBENCH_HEAT_3D::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N*m_N*m_N, vid);
  allocAndInitData(m_Binit, m_N*m_N*m_N, vid);
  allocAndInitDataConst(m_A, m_N*m_N*m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N*m_N*m_N, 0.0, vid);
}

void POLYBENCH_HEAT_3D::runKernel(VariantID vid)
{

  switch ( vid ) {

    case Base_Seq :
#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq :
    case RAJA_Seq :
#endif
    {
      runSeqVariant(vid);
      break;
    }

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
      runOpenMPVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_HEAT_3D::updateChecksum(VariantID vid)
{
  checksum[vid] += m_factor * calcChecksum(m_A, m_N*m_N*m_N);
  checksum[vid] += m_factor * calcChecksum(m_B, m_N*m_N*m_N);
}

void POLYBENCH_HEAT_3D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
