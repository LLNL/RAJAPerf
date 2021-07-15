//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_ATAX::POLYBENCH_ATAX(const RunParams& params)
  : KernelBase(rajaperf::Polybench_ATAX, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_N=42;
      run_reps = 10000;
      break;
    case Small:
      m_N=124;
      run_reps = 1000;
      break;
    case Medium:
      m_N=410;
      run_reps = 100;
      break;
    case Large:
      m_N=2100;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=2200;
      run_reps = 1;
      break;
    default:
      m_N=2100;
      run_reps = 100;
      break;
  }

#if 0 // we want this...

  Index_type N_default = 2100;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(100);

  m_N = std::sqrt( getTargetProblemSize() )+1;

#else  // this is what we have now...
  setDefaultProblemSize( m_N );
  setDefaultReps(run_reps);

#endif

  setActualProblemSize( m_N * m_N ); 

  setItsPerRep( m_N + m_N );
  setKernelsPerRep(2);
  setBytesPerRep( (2*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N * m_N +

                  (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N * m_N );
  setFLOPsPerRep(2 * m_N*m_N +
                 2 * m_N*m_N );

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

POLYBENCH_ATAX::~POLYBENCH_ATAX()
{
}

void POLYBENCH_ATAX::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_tmp, m_N, vid);
  allocAndInitData(m_x, m_N, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitDataConst(m_y, m_N, 0.0, vid);
}

void POLYBENCH_ATAX::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, m_N);
}

void POLYBENCH_ATAX::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_tmp);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_A);
}

} // end namespace polybench
} // end namespace rajaperf
