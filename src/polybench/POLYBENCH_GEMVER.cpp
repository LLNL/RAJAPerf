//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_GEMVER::POLYBENCH_GEMVER(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMVER, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_n=40;
      run_reps = 200;
      break;
    case Small:
      m_n=120;
      run_reps = 200;
      break;
    case Medium:
      m_n=400;
      run_reps = 20;
      break;
    case Large:
      m_n=2000;
      run_reps = 20;
      break;
    case Extralarge:
      m_n=4000;
      run_reps = 5;
      break;
    default:
      m_n=800;
      run_reps = 40;
      break;
  }

#if 0 // we want this...

  Index_type n_default = 1000;

  setDefaultProblemSize( n_default * n_default ) );
  setDefaultReps(20);

  m_n =  std::sqrt( getTargetProblemSize() ) + 1;

  m_alpha = 1.5;
  m_beta = 1.2;

#else // this is what we have now...

  m_alpha = 1.5;
  m_beta = 1.2;

  setDefaultProblemSize( m_n*m_n );
  setDefaultReps(run_reps);

#endif

  setActualProblemSize( m_n * m_n );

  setItsPerRep( m_n*m_n +
                m_n*m_n +
                m_n +
                m_n*m_n );
  setKernelsPerRep(4);
  setBytesPerRep( (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_n * m_n +
                  (0*sizeof(Real_type ) + 4*sizeof(Real_type )) * m_n +

                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_n * m_n +
                  (1*sizeof(Real_type ) + 2*sizeof(Real_type )) * m_n +

                  (1*sizeof(Real_type ) + 2*sizeof(Real_type )) * m_n +

                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_n * m_n +
                  (1*sizeof(Real_type ) + 2*sizeof(Real_type )) * m_n );
  setFLOPsPerRep(4 * m_n*m_n +
                 3 * m_n*m_n +
                 1 * m_n +
                 3 * m_n*m_n );

  setUsesFeature(Forall);
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

POLYBENCH_GEMVER::~POLYBENCH_GEMVER()
{
}

void POLYBENCH_GEMVER::setUp(VariantID vid)
{
  (void) vid;

  allocAndInitData(m_A, m_n * m_n, vid);
  allocAndInitData(m_u1, m_n, vid);
  allocAndInitData(m_v1, m_n, vid);
  allocAndInitData(m_u2, m_n, vid);
  allocAndInitData(m_v2, m_n, vid);
  allocAndInitDataConst(m_w, m_n, 0.0, vid);
  allocAndInitData(m_x, m_n, vid);
  allocAndInitData(m_y, m_n, vid);
  allocAndInitData(m_z, m_n, vid);
}

void POLYBENCH_GEMVER::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, m_n);
}

void POLYBENCH_GEMVER::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_u1);
  deallocData(m_v1);
  deallocData(m_u2);
  deallocData(m_v2);
  deallocData(m_w);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace basic
} // end namespace rajaperf
