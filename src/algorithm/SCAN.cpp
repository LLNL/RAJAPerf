//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SCAN.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


SCAN::SCAN(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_SCAN, params)
{
   setDefaultSize(100000);
   setDefaultReps(50);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( RAJA_CUDA );

  setVariantDefined( RAJA_HIP );
}

SCAN::~SCAN()
{
}

void SCAN::setUp(VariantID vid)
{
  allocAndInitDataRandValue(m_x, getRunSize(), vid);
  allocAndInitDataConst(m_y, getRunSize(), 0.0, vid);
}

void SCAN::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, getRunSize());
}

void SCAN::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace algorithm
} // end namespace rajaperf
