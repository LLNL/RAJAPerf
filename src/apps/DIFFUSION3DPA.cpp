//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{


DIFFUSION3DPA::DIFFUSION3DPA(const RunParams& params)
  : KernelBase(rajaperf::Apps_DIFFUSION3DPA, params)
{
  m_NE_default = 15625;

  setDefaultProblemSize(m_NE_default*DPA_Q1D*DPA_Q1D*DPA_Q1D);
  setDefaultReps(50);

  m_NE = std::max(getTargetProblemSize()/(DPA_Q1D*DPA_Q1D*DPA_Q1D), Index_type(1));

  setActualProblemSize( m_NE*DPA_Q1D*DPA_Q1D*DPA_Q1D );

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( 2*DPA_Q1D*DPA_D1D*sizeof(Real_type)  +
                  DPA_Q1D*DPA_Q1D*DPA_Q1D*SYM*m_NE*sizeof(Real_type) +
                  DPA_D1D*DPA_D1D*DPA_D1D*m_NE*sizeof(Real_type) +
                  DPA_D1D*DPA_D1D*DPA_D1D*m_NE*sizeof(Real_type) );

  setFLOPsPerRep(m_NE * (DPA_Q1D * DPA_D1D +
                         5 * DPA_D1D * DPA_D1D * DPA_Q1D * DPA_D1D +
                         7 * DPA_D1D * DPA_D1D * DPA_Q1D * DPA_Q1D +
                         7 * DPA_Q1D * DPA_D1D * DPA_Q1D * DPA_Q1D +
                         15 * DPA_Q1D * DPA_Q1D * DPA_Q1D +
                         DPA_Q1D * DPA_D1D +
                         7 * DPA_Q1D * DPA_Q1D * DPA_D1D * DPA_Q1D +
                         7 * DPA_Q1D * DPA_Q1D * DPA_D1D * DPA_D1D +
                         7 * DPA_D1D * DPA_Q1D * DPA_D1D * DPA_D1D +
                         3 * DPA_D1D * DPA_D1D * DPA_D1D));

  setUsesFeature(Teams);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
  setVariantDefined( RAJA_StdPar );
}

DIFFUSION3DPA::~DIFFUSION3DPA()
{
}

void DIFFUSION3DPA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  allocAndInitDataConst(m_B, int(DPA_Q1D*DPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_G, int(DPA_Q1D*DPA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(DPA_Q1D*DPA_Q1D*DPA_Q1D*SYM*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_X, int(DPA_D1D*DPA_D1D*DPA_D1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_Y, int(DPA_D1D*DPA_D1D*DPA_D1D*m_NE), Real_type(0.0), vid);
}

void DIFFUSION3DPA::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_Y, DPA_D1D*DPA_D1D*DPA_D1D*m_NE);
}

void DIFFUSION3DPA::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_B);
  deallocData(m_G);
  deallocData(m_D);
  deallocData(m_X);
  deallocData(m_Y);
}

} // end namespace apps
} // end namespace rajaperf
