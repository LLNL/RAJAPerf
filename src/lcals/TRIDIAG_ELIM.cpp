//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


TRIDIAG_ELIM::TRIDIAG_ELIM(const RunParams& params)
  : KernelBase(rajaperf::Lcals_TRIDIAG_ELIM, params)
{
   setDefaultSize(200000);
   setDefaultReps(5000);
}

TRIDIAG_ELIM::~TRIDIAG_ELIM() 
{
}

void TRIDIAG_ELIM::setUp(VariantID vid)
{
  m_N = getRunSize();

  allocAndInitDataConst(m_xout, m_N, 0.0, vid);
  allocAndInitData(m_xin, m_N, vid);
  allocAndInitData(m_y, m_N, vid);
  allocAndInitData(m_z, m_N, vid);
}

void TRIDIAG_ELIM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_xout, getRunSize());
}

void TRIDIAG_ELIM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_xout);
  deallocData(m_xin);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace lcals
} // end namespace rajaperf
