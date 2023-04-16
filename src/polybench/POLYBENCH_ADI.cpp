//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
  Index_type n_default = 1000;

  setDefaultProblemSize( (n_default-2) * (n_default-2) );
  setDefaultReps(4);

  m_n = std::sqrt( getTargetProblemSize() ) + 1;
  m_tsteps = 4;

  setItsPerRep( m_tsteps * ( (m_n-2) + (m_n-2) ) );


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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

POLYBENCH_ADI::~POLYBENCH_ADI()
{
}

void POLYBENCH_ADI::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_U, m_n * m_n, 0.0, vid);
  allocAndInitData(m_V, m_n * m_n, vid);
  allocAndInitData(m_P, m_n * m_n, vid);
  allocAndInitData(m_Q, m_n * m_n, vid);
}

void POLYBENCH_ADI::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_U, m_n * m_n, checksum_scale_factor , vid);
}

void POLYBENCH_ADI::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_U, vid);
  deallocData(m_V, vid);
  deallocData(m_P, vid);
  deallocData(m_Q, vid);
}

} // end namespace polybench
} // end namespace rajaperf
