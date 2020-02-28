//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace apps
{


FIR::FIR(const RunParams& params)
  : KernelBase(rajaperf::Apps_FIR, params)
{
  setDefaultSize(100000);
  setDefaultReps(1600);

  m_coefflen = FIR_COEFFLEN;
}

FIR::~FIR() 
{
}

Index_type FIR::getItsPerRep() const { 
  return getRunSize() - m_coefflen;
}

void FIR::setUp(VariantID vid)
{
  allocAndInitData(m_in, getRunSize(), vid);
  allocAndInitDataConst(m_out, getRunSize(), 0.0, vid);
}

void FIR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out, getRunSize());
}

void FIR::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_in);
  deallocData(m_out);
}

} // end namespace apps
} // end namespace rajaperf
