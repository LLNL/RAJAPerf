//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() - m_coefflen );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) * getItsPerRep() +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep((2 * m_coefflen) * (getActualProblemSize() - m_coefflen));

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
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

FIR::~FIR()
{
}

void FIR::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_in, getActualProblemSize(), vid);
  allocAndInitDataConst(m_out, getActualProblemSize(), 0.0, vid);
}

void FIR::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_out, getActualProblemSize(), checksum_scale_factor , vid);
}

void FIR::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_in, vid);
  deallocData(m_out, vid);
}

} // end namespace apps
} // end namespace rajaperf
