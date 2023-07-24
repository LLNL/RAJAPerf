//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY8.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


COPY8::COPY8(const RunParams& params)
  : KernelBase(rajaperf::Basic_COPY8, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (8*sizeof(Real_type) + 8*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(0);

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
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );
}

COPY8::~COPY8()
{
}

void COPY8::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_y0, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y1, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y2, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y3, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y4, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y5, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y6, getActualProblemSize(), 0.0, vid);
  allocAndInitDataConst(m_y7, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_x0, getActualProblemSize(), vid);
  allocAndInitData(m_x1, getActualProblemSize(), vid);
  allocAndInitData(m_x2, getActualProblemSize(), vid);
  allocAndInitData(m_x3, getActualProblemSize(), vid);
  allocAndInitData(m_x4, getActualProblemSize(), vid);
  allocAndInitData(m_x5, getActualProblemSize(), vid);
  allocAndInitData(m_x6, getActualProblemSize(), vid);
  allocAndInitData(m_x7, getActualProblemSize(), vid);
}

void COPY8::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_y0, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y1, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y2, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y3, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y4, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y5, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y6, getActualProblemSize(), vid);
  checksum[vid].at(tune_idx) += calcChecksum(m_y7, getActualProblemSize(), vid);
}

void COPY8::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x0, vid);
  deallocData(m_x1, vid);
  deallocData(m_x2, vid);
  deallocData(m_x3, vid);
  deallocData(m_x4, vid);
  deallocData(m_x5, vid);
  deallocData(m_x6, vid);
  deallocData(m_x7, vid);
  deallocData(m_y0, vid);
  deallocData(m_y1, vid);
  deallocData(m_y2, vid);
  deallocData(m_y3, vid);
  deallocData(m_y4, vid);
  deallocData(m_y5, vid);
  deallocData(m_y6, vid);
  deallocData(m_y7, vid);
}

} // end namespace basic
} // end namespace rajaperf
