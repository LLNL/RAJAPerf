//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


PI_ATOMIC::PI_ATOMIC(const RunParams& params)
  : KernelBase(rajaperf::Basic_PI_ATOMIC, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setItsPerRep( getRunProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) +
                  (0*sizeof(Real_type) + 0*sizeof(Real_type)) * getRunProblemSize() );
  setFLOPsPerRep(6 * getRunProblemSize() + 1);

  setUsesFeature(Forall);
  setUsesFeature(Atomic);

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

PI_ATOMIC::~PI_ATOMIC()
{
}

void PI_ATOMIC::setUp(VariantID vid)
{
  m_dx = 1.0 / double(getRunProblemSize());
  allocAndInitDataConst(m_pi, 1, 0.0, vid);
  m_pi_init = 0.0;
}

void PI_ATOMIC::updateChecksum(VariantID vid)
{
  checksum[vid] += Checksum_type(*m_pi);
}

void PI_ATOMIC::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_pi);
}

} // end namespace basic
} // end namespace rajaperf
