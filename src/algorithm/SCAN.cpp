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
   setDefaultSize(1000000);
   setDefaultReps(100);

  setProblemSize( getRunSize() );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getRunSize() );
  setFLOPsPerRep(1 * getRunSize());

  setUsesFeature(Scan);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

#if defined(_OPENMP) && _OPENMP >= 201811
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
#endif
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
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
