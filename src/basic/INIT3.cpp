//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


INIT3::INIT3(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT3, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setItsPerRep( getRunProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (3*sizeof(Real_type) + 2*sizeof(Real_type)) * getRunProblemSize() );
  setFLOPsPerRep(1 * getRunProblemSize());

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
}

INIT3::~INIT3()
{
}

void INIT3::setUp(VariantID vid)
{
  allocAndInitDataConst(m_out1, getRunProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_out2, getRunProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_out3, getRunProblemSize(), 0.0, vid);
  allocAndInitData(m_in1, getRunProblemSize(), vid);
  allocAndInitData(m_in2, getRunProblemSize(), vid);
}

void INIT3::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunProblemSize());
  checksum[vid] += calcChecksum(m_out2, getRunProblemSize());
  checksum[vid] += calcChecksum(m_out3, getRunProblemSize());
}

void INIT3::tearDown(VariantID vid)
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
