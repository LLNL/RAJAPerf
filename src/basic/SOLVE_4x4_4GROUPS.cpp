//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SOLVE_4x4_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


SOLVE_4x4_4GROUPS::SOLVE_4x4_4GROUPS(const RunParams& params)
  : KernelBase(rajaperf::Basic_SOLVE_4x4_4GROUPS, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( ( ( 16*4 + 4*4 ) * sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep( 61 * 4 * getActualProblemSize());

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

}

SOLVE_4x4_4GROUPS::~SOLVE_4x4_4GROUPS()
{
}

void SOLVE_4x4_4GROUPS::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataRandValue     (m_a, getActualProblemSize()*16*4,      vid);
  allocAndInitDataRandValue     (m_x, getActualProblemSize()* 4*4,      vid);
  allocAndInitDataConst         (m_y, getActualProblemSize()* 4*4,      vid);
}

void SOLVE_4x4_4GROUPS::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_y, getActualProblemSize()*4*4);
}

void SOLVE_4x4_4GROUPS::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace basic
} // end namespace rajaperf
