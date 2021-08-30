//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


IF_QUAD::IF_QUAD(const RunParams& params)
  : KernelBase(rajaperf::Basic_IF_QUAD, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(180);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() ); 
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type) + 3*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(11 * getActualProblemSize()); // 1 sqrt

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
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
  setVariantDefined( RAJA_StdPar );
}

IF_QUAD::~IF_QUAD()
{
}

void IF_QUAD::setUp(VariantID vid)
{
  allocAndInitDataRandSign(m_a, getActualProblemSize(), vid);
  allocAndInitData(m_b, getActualProblemSize(), vid);
  allocAndInitData(m_c, getActualProblemSize(), vid);
  allocAndInitDataConst(m_x1, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_x2, getActualProblemSize(), 0.0, vid);
}

void IF_QUAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x1, getActualProblemSize(), checksum_scale_factor );
  checksum[vid] += calcChecksum(m_x2, getActualProblemSize(), checksum_scale_factor );
}

void IF_QUAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
  deallocData(m_x1);
  deallocData(m_x2);
}

} // end namespace basic
} // end namespace rajaperf
