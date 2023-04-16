//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"
namespace rajaperf
{
namespace lcals
{


DIFF_PREDICT::DIFF_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_DIFF_PREDICT, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(200);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );

  setKernelsPerRep(1);
  setBytesPerRep( (10*sizeof(Real_type) + 10*sizeof(Real_type)) * getActualProblemSize());
  setFLOPsPerRep(9 * getActualProblemSize());

  setUsesFeature(Forall);

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

  setVariantDefined( Kokkos_Lambda );

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

DIFF_PREDICT::~DIFF_PREDICT()
{
}

void DIFF_PREDICT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  m_array_length = getActualProblemSize() * 14;
  m_offset = getActualProblemSize();

  allocAndInitDataConst(m_px, m_array_length, 0.0, vid);
  allocAndInitData(m_cx, m_array_length, vid);
}

void DIFF_PREDICT::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_px, m_array_length, vid);
}

void DIFF_PREDICT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_px, vid);
  deallocData(m_cx, vid);
}

} // end namespace lcals
} // end namespace rajaperf
