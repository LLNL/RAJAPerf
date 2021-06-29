//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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
  setDefaultSize(1000000);
  setDefaultReps(130);

  setProblemSize( getRunSize() );

  setItsPerRep( 6 * getProblemSize() );
  setKernelsPerRep(1);
  // some branches are never taken due to the nature of the initialization of delvc
  // the additional reads and writes that would be done if those branches were taken are noted in the comments
  setBytesPerRep( (1*sizeof(Real_type) + 5*sizeof(Real_type)) * getRunSize() +
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunSize() + /* 1 + 8 */
                  (1*sizeof(Real_type) + 6*sizeof(Real_type)) * getRunSize() +
                  (1*sizeof(Real_type) + 2*sizeof(Real_type)) * getRunSize() +
                  (1*sizeof(Real_type) + 7*sizeof(Real_type)) * getRunSize() + /* 1 + 12 */
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunSize() ); /* 1 + 8 */
  setFLOPsPerRep((6  +
                  11 + // 1 sqrt
                  8  +
                  2  +
                  19 + // 1 sqrt
                  9    // 1 sqrt
                  ) * getProblemSize());

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

ENERGY::~ENERGY()
{
}

void ENERGY::setUp(VariantID vid)
{
  allocAndInitDataConst(m_e_new, getRunSize(), 0.0, vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_delvc, getRunSize(), vid);
  allocAndInitData(m_p_new, getRunSize(), vid);
  allocAndInitData(m_p_old, getRunSize(), vid);
  allocAndInitDataConst(m_q_new, getRunSize(), 0.0, vid);
  allocAndInitData(m_q_old, getRunSize(), vid);
  allocAndInitData(m_work, getRunSize(), vid);
  allocAndInitData(m_compHalfStep, getRunSize(), vid);
  allocAndInitData(m_pHalfStep, getRunSize(), vid);
  allocAndInitData(m_bvc, getRunSize(), vid);
  allocAndInitData(m_pbvc, getRunSize(), vid);
  allocAndInitData(m_ql_old, getRunSize(), vid);
  allocAndInitData(m_qq_old, getRunSize(), vid);
  allocAndInitData(m_vnewc, getRunSize(), vid);

  initData(m_rho0);
  initData(m_e_cut);
  initData(m_emin);
  initData(m_q_cut);
}

void ENERGY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_e_new, getRunSize());
  checksum[vid] += calcChecksum(m_q_new, getRunSize());
}

void ENERGY::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_e_new);
  deallocData(m_e_old);
  deallocData(m_delvc);
  deallocData(m_p_new);
  deallocData(m_p_old);
  deallocData(m_q_new);
  deallocData(m_q_old);
  deallocData(m_work);
  deallocData(m_compHalfStep);
  deallocData(m_pHalfStep);
  deallocData(m_bvc);
  deallocData(m_pbvc);
  deallocData(m_ql_old);
  deallocData(m_qq_old);
  deallocData(m_vnewc);
}

} // end namespace apps
} // end namespace rajaperf
