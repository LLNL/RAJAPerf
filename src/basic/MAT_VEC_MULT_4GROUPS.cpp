//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_VEC_MULT_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


MAT_VEC_MULT_4GROUPS::MAT_VEC_MULT_4GROUPS(const RunParams& params)
  : KernelBase(rajaperf::Basic_MAT_VEC_MULT_4GROUPS, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( ( 3 * 16 * sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep( 4 * 4 * 7 * getActualProblemSize());

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

}

MAT_VEC_MULT_4GROUPS::~MAT_VEC_MULT_4GROUPS()
{
}

void MAT_VEC_MULT_4GROUPS::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_y, getActualProblemSize()*16, 0.0, vid);
  allocAndInitData(m_x, getActualProblemSize()*16, vid);
  allocAndInitData(m_a, getActualProblemSize()*16, vid);
}

void MAT_VEC_MULT_4GROUPS::updateChecksum(VariantID vid, size_t tune_idx)
{
  Real_type sum = calcChecksum(m_y, getActualProblemSize()*16);
  checksum[vid].at(tune_idx) += sum;
}

void MAT_VEC_MULT_4GROUPS::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace basic
} // end namespace rajaperf
