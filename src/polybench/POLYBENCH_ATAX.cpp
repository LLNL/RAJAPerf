//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
  Index_type N_default = 1000;

  setDefaultProblemSize( N_default * N_default );
  setDefaultReps(100);

  m_N = std::sqrt( getTargetProblemSize() )+1;


  setActualProblemSize( m_N * m_N ); 

  setItsPerRep( m_N + m_N );
  setKernelsPerRep(2);
  setBytesPerRep( (2*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N * m_N +

                  (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_N * m_N );
  setFLOPsPerRep(2 * m_N*m_N +
                 2 * m_N*m_N );

  checksum_scale_factor = 0.001 * 
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
  checksum[vid] += calcChecksum(m_y, m_N, checksum_scale_factor );
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
