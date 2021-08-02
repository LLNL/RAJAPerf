//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


MULADDSUB::MULADDSUB(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULADDSUB, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(350);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (3*sizeof(Real_type) + 2*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(3 * getActualProblemSize());

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

MULADDSUB::~MULADDSUB()
{
}

void MULADDSUB::setUp(VariantID vid)
{
  allocAndInitDataConst(m_out1, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_out2, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_out3, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_in1, getActualProblemSize(), vid);
  allocAndInitData(m_in2, getActualProblemSize(), vid);
}

void MULADDSUB::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getActualProblemSize());
  checksum[vid] += calcChecksum(m_out2, getActualProblemSize());
  checksum[vid] += calcChecksum(m_out3, getActualProblemSize());
}

void MULADDSUB::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_out1);
  deallocData(m_out2);
  deallocData(m_out3);
  deallocData(m_in1);
  deallocData(m_in2);
}

} // end namespace basic
} // end namespace rajaperf
