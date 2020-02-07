//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace basic
{


DAXPY::DAXPY(const RunParams& params)
  : KernelBase(rajaperf::Basic_DAXPY, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

DAXPY::~DAXPY() 
{
}

void DAXPY::setUp(VariantID vid)
{
  allocAndInitDataConst(m_y, getRunSize(), 0.0, vid);
  allocAndInitData(m_x, getRunSize(), vid);
  initData(m_a);
}

void DAXPY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, getRunSize());
}

void DAXPY::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace basic
} // end namespace rajaperf
