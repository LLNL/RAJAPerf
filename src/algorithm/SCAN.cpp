//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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
  setDefaultProblemSize(1000000);
  setDefaultReps(100);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(1 * getActualProblemSize());

  checksum_scale_factor = 1e-2 *
                 ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                              getActualProblemSize() ) /
                 getActualProblemSize();

  setUsesFeature(Scan);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

#if _OPENMP >= 201811 && defined(RAJA_PERFSUITE_ENABLE_OPENMP5_SCAN)
  setVariantDefined( Base_OpenMPTarget );
#endif

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
}

SCAN::~SCAN()
{
}

void SCAN::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataRandValue(m_x, getActualProblemSize(), vid);
  allocAndInitDataConst(m_y, getActualProblemSize(), 0.0, vid);
}

void SCAN::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_y, getActualProblemSize(), checksum_scale_factor, vid);
}

void SCAN::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x, vid);
  deallocData(m_y, vid);
}

} // end namespace algorithm
} // end namespace rajaperf
