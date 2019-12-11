//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_DIFF.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


FIRST_DIFF::FIRST_DIFF(const RunParams& params)
  : KernelBase(rajaperf::Lcals_FIRST_DIFF, params)
{
   setDefaultSize(100000);
   setDefaultReps(16000);
}

FIRST_DIFF::~FIRST_DIFF() 
{
}

void FIRST_DIFF::setUp(VariantID vid)
{
  m_array_length = getRunSize()+1; 
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitData(m_y, m_array_length, vid);
}

void FIRST_DIFF::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void FIRST_DIFF::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace lcals
} // end namespace rajaperf
