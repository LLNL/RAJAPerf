//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


COPY::COPY(const RunParams& params)
  : KernelBase(rajaperf::Stream_COPY, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1800);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * 
                  getActualProblemSize() );
  setFLOPsPerRep(0);

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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
  setVariantDefined( RAJA_StdPar );
}

COPY::~COPY()
{
}

void COPY::setUp(VariantID vid)
{
  allocAndInitData(m_a, getActualProblemSize(), vid);
  allocAndInitDataConst(m_c, getActualProblemSize(), 0.0, vid);
}

void COPY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getActualProblemSize());
}

void COPY::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
