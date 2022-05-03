//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{


CONVECTION3DPA::CONVECTION3DPA(const RunParams& params)
  : KernelBase(rajaperf::Apps_CONVECTION3DPA, params)
{
  m_NE_default = 15625;

  setDefaultProblemSize(m_NE_default*CPA_Q1D*CPA_Q1D*CPA_Q1D);
  setDefaultReps(50);

  m_NE = std::max(getTargetProblemSize()/(CPA_Q1D*CPA_Q1D*CPA_Q1D), Index_type(1));

  setActualProblemSize( m_NE*CPA_Q1D*CPA_Q1D*CPA_Q1D );

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( 3*CPA_Q1D*CPA_D1D*sizeof(Real_type)  +
                  CPA_VDIM*CPA_Q1D*CPA_Q1D*CPA_Q1D*m_NE*sizeof(Real_type) +
                  CPA_D1D*CPA_D1D*CPA_D1D*m_NE*sizeof(Real_type) +
                  CPA_D1D*CPA_D1D*CPA_D1D*m_NE*sizeof(Real_type) );

  setFLOPsPerRep(m_NE * (
                         4 * CPA_D1D * CPA_Q1D * CPA_D1D * CPA_D1D + //2
                         6 * CPA_D1D * CPA_Q1D * CPA_Q1D * CPA_D1D + //3
                         6 * CPA_D1D * CPA_Q1D * CPA_Q1D * CPA_Q1D + //4
                         5 * CPA_Q1D * CPA_Q1D * CPA_Q1D +  // 5
                         2 * CPA_Q1D * CPA_D1D * CPA_Q1D * CPA_Q1D + // 6
                         2 * CPA_Q1D * CPA_D1D * CPA_Q1D * CPA_D1D + // 7
                         (1 + 2*CPA_Q1D) * CPA_D1D * CPA_D1D * CPA_D1D // 8
                         ));

  setUsesFeature(Teams);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

}

CONVECTION3DPA::~CONVECTION3DPA()
{
}

void CONVECTION3DPA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  allocAndInitDataConst(m_B,  int(CPA_Q1D*CPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_Bt, int(CPA_Q1D*CPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_G, int(CPA_Q1D*CPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(CPA_Q1D*CPA_Q1D*CPA_Q1D*CPA_VDIM*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_X, int(CPA_D1D*CPA_D1D*CPA_D1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_Y, int(CPA_D1D*CPA_D1D*CPA_D1D*m_NE), Real_type(0.0), vid);
}

void CONVECTION3DPA::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_Y, CPA_D1D*CPA_D1D*CPA_D1D*m_NE);
}

void CONVECTION3DPA::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_B);
  deallocData(m_Bt);
  deallocData(m_G);
  deallocData(m_D);
  deallocData(m_X);
  deallocData(m_Y);
}

} // end namespace apps
} // end namespace rajaperf
