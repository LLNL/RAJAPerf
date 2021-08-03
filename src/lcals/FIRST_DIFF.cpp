//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


FIRST_DIFF::FIRST_DIFF(const RunParams& params)
  : KernelBase(rajaperf::Lcals_FIRST_DIFF, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(2000);

  setActualProblemSize( getTargetProblemSize() );

  m_N = getActualProblemSize()+1;

  setItsPerRep( getActualProblemSize() );
  setItsPerRep( getActualProblemSize() ); // why twice?
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) * getActualProblemSize() +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * m_N );
  setFLOPsPerRep(1 * getActualProblemSize());

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
  setVariantDefined( RAJA_StdPar );
}

FIRST_DIFF::~FIRST_DIFF()
{
}

void FIRST_DIFF::setUp(VariantID vid)
{
  allocAndInitDataConst(m_x, m_N, 0.0, vid);
  allocAndInitData(m_y, m_N, vid);
}

void FIRST_DIFF::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getActualProblemSize());
}

void FIRST_DIFF::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace lcals
} // end namespace rajaperf
