//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_JACOBI_2D::POLYBENCH_JACOBI_2D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_2D, params)
{
  Index_type N_default = 1000;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(50);

  m_N = std::sqrt( getTargetProblemSize() ) + 1;
  m_tsteps = 40;


  setActualProblemSize( (m_N-2) * (m_N-2) );

  setItsPerRep( m_tsteps * (2 * (m_N-2) * (m_N-2)) );
  setKernelsPerRep(2);
  setBytesPerRep( m_tsteps * ( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) * (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               (m_N * m_N - 4) +
                               (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) * (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               (m_N * m_N  - 4) ) );
  setFLOPsPerRep( m_tsteps * ( 5 * (m_N-2)*(m_N-2) +
                               5 * (m_N -2)*(m_N-2) ) );

  checksum_scale_factor = 0.0001 *
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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
  setVariantDefined( RAJA_StdPar );
}

POLYBENCH_JACOBI_2D::~POLYBENCH_JACOBI_2D()
{
}

void POLYBENCH_JACOBI_2D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N*m_N, vid);
  allocAndInitData(m_Binit, m_N*m_N, vid);
  allocAndInitDataConst(m_A, m_N*m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N*m_N, 0.0, vid);
}

void POLYBENCH_JACOBI_2D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_A, m_N*m_N, checksum_scale_factor );
  checksum[vid][tune_idx] += calcChecksum(m_B, m_N*m_N, checksum_scale_factor );
}

void POLYBENCH_JACOBI_2D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
