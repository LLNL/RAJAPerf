//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( 2 * getActualProblemSize() );
  setKernelsPerRep(2);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() +
                  (1*sizeof(Real_type) + 2*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep((2 +
                  1
                  ) * getActualProblemSize());

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

PRESSURE::~PRESSURE()
{
}

void PRESSURE::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_compression, getActualProblemSize(), vid);
  allocAndInitData(m_bvc, getActualProblemSize(), vid);
  allocAndInitDataConst(m_p_new, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_e_old, getActualProblemSize(), vid);
  allocAndInitData(m_vnewc, getActualProblemSize(), vid);

  initData(m_cls, vid);
  initData(m_p_cut, vid);
  initData(m_pmin, vid);
  initData(m_eosvmax, vid);
}

void PRESSURE::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_p_new, getActualProblemSize(), vid);
}

void PRESSURE::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_compression, vid);
  deallocData(m_bvc, vid);
  deallocData(m_p_new, vid);
  deallocData(m_e_old, vid);
  deallocData(m_vnewc, vid);
}

} // end namespace apps
} // end namespace rajaperf
