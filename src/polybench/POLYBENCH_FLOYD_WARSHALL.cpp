//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_FLOYD_WARSHALL::POLYBENCH_FLOYD_WARSHALL(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FLOYD_WARSHALL, params)
{
  Index_type N_default = 1000;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(8);

  m_N = std::sqrt( getTargetProblemSize() ) + 1;


  setActualProblemSize( m_N * m_N );

  setItsPerRep( m_N*m_N );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N * m_N );
  setFLOPsPerRep(1 * m_N*m_N*m_N );

  checksum_scale_factor = 1.0 *
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

POLYBENCH_FLOYD_WARSHALL::~POLYBENCH_FLOYD_WARSHALL()
{
}

void POLYBENCH_FLOYD_WARSHALL::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitDataRandSign(m_pin, m_N*m_N, vid);
  allocAndInitDataConst(m_pout, m_N*m_N, 0.0, vid);
}

void POLYBENCH_FLOYD_WARSHALL::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_pout, m_N*m_N, checksum_scale_factor , vid);
}

void POLYBENCH_FLOYD_WARSHALL::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_pin, vid);
  deallocData(m_pout, vid);
}

} // end namespace polybench
} // end namespace rajaperf
