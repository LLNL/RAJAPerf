//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_JACOBI_1D::POLYBENCH_JACOBI_1D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_1D, params)
{
  Index_type N_default = 1000000;

  setDefaultProblemSize( N_default-2 );
  setDefaultReps(100);

  m_N = getTargetProblemSize();
  m_tsteps = 16;


  setActualProblemSize( m_N-2 );

  setItsPerRep( m_tsteps * ( 2 * getActualProblemSize() ) );
  setKernelsPerRep(m_tsteps * 2);
  setBytesPerRep( m_tsteps * ( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               m_N +
                               (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               m_N ) );
  setFLOPsPerRep( m_tsteps * ( 3 * (m_N-2) +
                               3 * (m_N-2) ) );

  checksum_scale_factor = 0.0001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Forall);

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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

POLYBENCH_JACOBI_1D::~POLYBENCH_JACOBI_1D()
{
}

void POLYBENCH_JACOBI_1D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N, vid);
  allocAndInitData(m_Binit, m_N, vid);
  allocData(m_A, m_N, vid);
  allocData(m_B, m_N, vid);
}

void POLYBENCH_JACOBI_1D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_A, m_N, checksum_scale_factor , vid);
  checksum[vid][tune_idx] += calcChecksum(m_B, m_N, checksum_scale_factor , vid);
}

void POLYBENCH_JACOBI_1D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_Ainit, vid);
  deallocData(m_Binit, vid);
}

} // end namespace polybench
} // end namespace rajaperf
