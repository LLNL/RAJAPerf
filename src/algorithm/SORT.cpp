//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
  setDefaultProblemSize(1000000);
  setDefaultReps(20);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() ); // touched data size, not actual number of stores and loads
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

void SORT::setUp(VariantID vid, size_t /*tid*/)
{
  allocAndInitDataRandValue(m_x, getActualProblemSize()*getRunReps(), vid);
}

void SORT::updateChecksum(VariantID vid, size_t tid)
{
  checksum[vid][tid] += calcChecksum(m_x, getActualProblemSize()*getRunReps());
}

void SORT::tearDown(VariantID vid, size_t /*tid*/)
{
  (void) vid;
  deallocData(m_x);
}

} // end namespace algorithm
} // end namespace rajaperf
