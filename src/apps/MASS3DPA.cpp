//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASSPA3D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace apps
{


MASSPA3D::MASSPA3D(const RunParams& params)
  : KernelBase(rajaperf::Apps_MASSPA3D, params)
{
  m_NE_default = 924385;

  setDefaultSize(m_NE_default);

  setDefaultReps(50);
}

MASSPA3D::~MASSPA3D() 
{
}

void MASSPA3D::setUp(VariantID vid)
{
  m_NE = run_params.getSizeFactor() * m_NE_default;     

  allocAndInitDataConst(m_B, int(m_Q1D*m_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_Bt,int(m_Q1D*m_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(m_Q1D*m_Q1D*m_Q1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_X, int(m_D1D*m_D1D*m_D1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_Y, int(m_D1D*m_D1D*m_D1D*m_NE), Real_type(0.0), vid);
}

void MASSPA3D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_Y, m_D1D*m_D1D*m_D1D*NE);
}

void MASSPA3D::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_B);
  deallocData(m_Bt);
  deallocData(m_D);
  deallocData(m_X);
  deallocData(m_Y);
}

} // end namespace apps
} // end namespace rajaperf
