//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


ENERGY::ENERGY(const RunParams& params)
  : KernelBase(rajaperf::Apps_ENERGY, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(130);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( 6 * getActualProblemSize() );
  setKernelsPerRep(6);
  // some branches are never taken due to the nature of the initialization of delvc
  // the additional reads and writes that would be done if those branches were taken are noted in the comments
  setBytesPerRep( (1*sizeof(Real_type) + 5*sizeof(Real_type)) * getActualProblemSize() +
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() + /* 1 + 8 */
                  (1*sizeof(Real_type) + 6*sizeof(Real_type)) * getActualProblemSize() +
                  (1*sizeof(Real_type) + 2*sizeof(Real_type)) * getActualProblemSize() +
                  (1*sizeof(Real_type) + 7*sizeof(Real_type)) * getActualProblemSize() + /* 1 + 12 */
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() ); /* 1 + 8 */
  setFLOPsPerRep((6  +
                  11 + // 1 sqrt
                  8  +
                  2  +
                  19 + // 1 sqrt
                  9    // 1 sqrt
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

ENERGY::~ENERGY()
{
}

void ENERGY::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_e_new, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_e_old, getActualProblemSize(), vid);
  allocAndInitData(m_delvc, getActualProblemSize(), vid);
  allocAndInitData(m_p_new, getActualProblemSize(), vid);
  allocAndInitData(m_p_old, getActualProblemSize(), vid);
  allocAndInitDataConst(m_q_new, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_q_old, getActualProblemSize(), vid);
  allocAndInitData(m_work, getActualProblemSize(), vid);
  allocAndInitData(m_compHalfStep, getActualProblemSize(), vid);
  allocAndInitData(m_pHalfStep, getActualProblemSize(), vid);
  allocAndInitData(m_bvc, getActualProblemSize(), vid);
  allocAndInitData(m_pbvc, getActualProblemSize(), vid);
  allocAndInitData(m_ql_old, getActualProblemSize(), vid);
  allocAndInitData(m_qq_old, getActualProblemSize(), vid);
  allocAndInitData(m_vnewc, getActualProblemSize(), vid);

  initData(m_rho0, vid);
  initData(m_e_cut, vid);
  initData(m_emin, vid);
  initData(m_q_cut, vid);
}

void ENERGY::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_e_new, getActualProblemSize(), vid);
  checksum[vid][tune_idx] += calcChecksum(m_q_new, getActualProblemSize(), vid);
}

void ENERGY::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_e_new, vid);
  deallocData(m_e_old, vid);
  deallocData(m_delvc, vid);
  deallocData(m_p_new, vid);
  deallocData(m_p_old, vid);
  deallocData(m_q_new, vid);
  deallocData(m_q_old, vid);
  deallocData(m_work, vid);
  deallocData(m_compHalfStep, vid);
  deallocData(m_pHalfStep, vid);
  deallocData(m_bvc, vid);
  deallocData(m_pbvc, vid);
  deallocData(m_ql_old, vid);
  deallocData(m_qq_old, vid);
  deallocData(m_vnewc, vid);
}

} // end namespace apps
} // end namespace rajaperf
