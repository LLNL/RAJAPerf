//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace basic
{


INIT_VIEW1D_OFFSET::INIT_VIEW1D_OFFSET(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT_VIEW1D_OFFSET, params)
{
   setDefaultSize(500000);
   setDefaultReps(5000);
}

INIT_VIEW1D_OFFSET::~INIT_VIEW1D_OFFSET() 
{
}

void INIT_VIEW1D_OFFSET::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  m_val = 0.123;  
}

void INIT_VIEW1D_OFFSET::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void INIT_VIEW1D_OFFSET::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
}

} // end namespace basic
} // end namespace rajaperf
