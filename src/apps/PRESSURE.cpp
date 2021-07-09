//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


PRESSURE::PRESSURE(const RunParams& params)
  : KernelBase(rajaperf::Apps_PRESSURE, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(700);

  setItsPerRep( 2 * getRunProblemSize() );
  setKernelsPerRep(2);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunProblemSize() +
                  (1*sizeof(Real_type) + 2*sizeof(Real_type)) * getRunProblemSize() );
  setFLOPsPerRep((2 +
                  1
                  ) * getRunProblemSize());

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

PRESSURE::~PRESSURE()
{
}

void PRESSURE::setUp(VariantID vid)
{
  allocAndInitData(m_compression, getRunProblemSize(), vid);
  allocAndInitData(m_bvc, getRunProblemSize(), vid);
  allocAndInitDataConst(m_p_new, getRunProblemSize(), 0.0, vid);
  allocAndInitData(m_e_old, getRunProblemSize(), vid);
  allocAndInitData(m_vnewc, getRunProblemSize(), vid);

  initData(m_cls);
  initData(m_p_cut);
  initData(m_pmin);
  initData(m_eosvmax);
}

void PRESSURE::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_p_new, getRunProblemSize());
}

void PRESSURE::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_compression);
  deallocData(m_bvc);
  deallocData(m_p_new);
  deallocData(m_e_old);
  deallocData(m_vnewc);
}

} // end namespace apps
} // end namespace rajaperf
