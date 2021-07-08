//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


TRIAD::TRIAD(const RunParams& params)
  : KernelBase(rajaperf::Stream_TRIAD, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1000);

  setTargetProblemSize( getRunProblemSize() );

  setItsPerRep( getRunProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 2*sizeof(Real_type)) * 
                  getRunProblemSize() );
  setFLOPsPerRep(2 * getRunProblemSize());

  setUsesFeature( Forall );

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

TRIAD::~TRIAD()
{
}

void TRIAD::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunProblemSize(), 0.0, vid);
  allocAndInitData(m_b, getRunProblemSize(), vid);
  allocAndInitData(m_c, getRunProblemSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunProblemSize());
}

void TRIAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
