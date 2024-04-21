//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <cmath>

namespace rajaperf
{
namespace polybench
{


POLYBENCH_HEAT_3D::POLYBENCH_HEAT_3D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_HEAT_3D, params)
{
  Index_type N_default = 100;

  setDefaultProblemSize( (N_default-2)*(N_default-2)*(N_default-2) );
  setDefaultReps(20);

  m_N = std::cbrt( getTargetProblemSize() ) + 1;
  m_tsteps = 20;


  setActualProblemSize( (m_N-2) * (m_N-2) * (m_N-2) );

  setItsPerRep( m_tsteps * ( 2 * getActualProblemSize() ) );
  setKernelsPerRep( m_tsteps * 2 );
  setBytesPerRep( m_tsteps * ( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) * (m_N-2) * (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               (m_N * m_N * m_N - 12*(m_N-2) - 8) +
                               (1*sizeof(Real_type ) + 0*sizeof(Real_type )) *
                               (m_N-2) * (m_N-2) * (m_N-2) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) *
                               (m_N * m_N * m_N - 12*(m_N-2) - 8) ) );
  setFLOPsPerRep( m_tsteps * ( 15 * (m_N-2) * (m_N-2) * (m_N-2) +
                               15 * (m_N-2) * (m_N-2) * (m_N-2) ) );

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
}

POLYBENCH_HEAT_3D::~POLYBENCH_HEAT_3D()
{
}

void POLYBENCH_HEAT_3D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N*m_N*m_N, vid);
  allocAndInitData(m_Binit, m_N*m_N*m_N, vid);
  allocData(m_A, m_N*m_N*m_N, vid);
  allocData(m_B, m_N*m_N*m_N, vid);
}

void POLYBENCH_HEAT_3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_A, m_N*m_N*m_N, checksum_scale_factor , vid);
  checksum[vid][tune_idx] += calcChecksum(m_B, m_N*m_N*m_N, checksum_scale_factor , vid);
}

void POLYBENCH_HEAT_3D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_Ainit, vid);
  deallocData(m_Binit, vid);
}

} // end namespace polybench
} // end namespace rajaperf
