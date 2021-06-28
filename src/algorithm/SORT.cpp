//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


SORT::SORT(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_SORT, params)
{
  setDefaultSize(1000000);
  setDefaultReps(20);

  setProblemSize( getRunSize() );
   
  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setFLOPsPerRep(0);

  setUsesFeature(Sort);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( RAJA_CUDA );

  setVariantDefined( RAJA_HIP );
}

SORT::~SORT()
{
}

size_t SORT::getBytesPerRep() const
{
  return (1*sizeof(Index_type) + 1*sizeof(Index_type)) * getRunSize() ; // touched data size, not actual number of stores and loads
}

void SORT::setUp(VariantID vid)
{
  allocAndInitDataRandValue(m_x, getRunSize()*getRunReps(), vid);
}

void SORT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize()*getRunReps());
}

void SORT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
}

} // end namespace algorithm
} // end namespace rajaperf
