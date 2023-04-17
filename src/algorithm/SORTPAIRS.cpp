//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


SORTPAIRS::SORTPAIRS(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_SORTPAIRS, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(20);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type) + 2*sizeof(Real_type)) * getActualProblemSize() ); // touched data size, not actual number of stores and loads
  setFLOPsPerRep(0);

  setUsesFeature(Sort);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( RAJA_CUDA );

  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
}

SORTPAIRS::~SORTPAIRS()
{
}

void SORTPAIRS::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataRandValue(m_x, getActualProblemSize()*getRunReps(), vid);
  allocAndInitDataRandValue(m_i, getActualProblemSize()*getRunReps(), vid);
}

void SORTPAIRS::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_x, getActualProblemSize()*getRunReps(), vid);
  checksum[vid][tune_idx] += calcChecksum(m_i, getActualProblemSize()*getRunReps(), vid);
}

void SORTPAIRS::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x, vid);
  deallocData(m_i, vid);
}

} // end namespace algorithm
} // end namespace rajaperf
