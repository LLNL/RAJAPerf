//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


FIR::FIR(const RunParams& params)
  : KernelBase(rajaperf::Apps_FIR, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(160);

  m_coefflen = FIR_COEFFLEN;

  setItsPerRep( getRunProblemSize() - m_coefflen );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) * getItsPerRep() +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunProblemSize() );
  setFLOPsPerRep((2 * m_coefflen) * (getRunProblemSize() - m_coefflen));

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
}

FIR::~FIR()
{
}

void FIR::setUp(VariantID vid)
{
  allocAndInitData(m_in, getRunProblemSize(), vid);
  allocAndInitDataConst(m_out, getRunProblemSize(), 0.0, vid);
}

void FIR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out, getRunProblemSize());
}

void FIR::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_in);
  deallocData(m_out);
}

} // end namespace apps
} // end namespace rajaperf
