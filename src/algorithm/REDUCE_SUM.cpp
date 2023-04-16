//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


REDUCE_SUM::REDUCE_SUM(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_REDUCE_SUM, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(getActualProblemSize());

  setUsesFeature(Forall);
  setUsesFeature(Reduction);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
  //setVariantDefined( Lambda_StdPar ); // exists but is not interesting
}

REDUCE_SUM::~REDUCE_SUM()
{
}

void REDUCE_SUM::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_x, getActualProblemSize(), vid);
  m_sum_init = 0.0;
  m_sum = 0.0;
}

void REDUCE_SUM::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(&m_sum, 1, vid);
}

void REDUCE_SUM::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x, vid);
}

} // end namespace algorithm
} // end namespace rajaperf
