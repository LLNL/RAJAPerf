//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


ADD_4GROUPS::ADD_4GROUPS(const RunParams& params)
  : KernelBase(rajaperf::Basic_ADD_4GROUPS, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( ( ( 16*4*2 + 16 ) * sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep( 4 * 4 * 7 * getActualProblemSize());

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );


  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

}

ADD_4GROUPS::~ADD_4GROUPS()
{
}

void ADD_4GROUPS::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData     (m_a, getActualProblemSize()*16*4,      vid);
  allocAndInitData     (m_b, getActualProblemSize()*16  ,      vid);
  allocAndInitDataConst(m_c, getActualProblemSize()*16*4, 0.0, vid);
}

void ADD_4GROUPS::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_c, getActualProblemSize());
}

void ADD_4GROUPS::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace basic
} // end namespace rajaperf
