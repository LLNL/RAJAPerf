//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{

MASS3DPA::MASS3DPA(const RunParams& params)
  : KernelBase(rajaperf::Apps_MASS3DPA, params)
{
  Index_type NE_default = 8000;
  setDefaultProblemSize(NE_default*mpa::D1D*mpa::D1D*mpa::D1D);
  setDefaultReps(50);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(2);
  setProblemDimensionality(3);

  setUsesFeature(Launch);

  addVariantTunings();
}

void MASS3DPA::setSize(Index_type target_size, Index_type target_reps)
{
  m_NE = std::max((target_size + (mpa::D1D*mpa::D1D*mpa::D1D)/2) / (mpa::D1D*mpa::D1D*mpa::D1D), Index_type(1));

  setActualProblemSize( m_NE*mpa::D1D*mpa::D1D*mpa::D1D );
  setRunReps( target_reps );

  setItsPerRep( m_NE*mpa::D1D*mpa::D1D );
  setKernelsPerRep(1);

  setBytesAllocatedPerRep( 2*sizeof(Real_type) * mpa::Q1D*mpa::D1D + // B, Bt
                           1*sizeof(Real_type) * mpa::Q1D*mpa::Q1D*mpa::Q1D*m_NE + // D
                           2*sizeof(Real_type) * mpa::D1D*mpa::D1D*mpa::D1D*m_NE ); // X, Y
  setBytesReadPerRep( 2*sizeof(Real_type) * mpa::Q1D*mpa::D1D + // B, Bt
                      1*sizeof(Real_type) * mpa::D1D*mpa::D1D*mpa::D1D*m_NE + // X
                      1*sizeof(Real_type) * mpa::Q1D*mpa::Q1D*mpa::Q1D*m_NE ); // D
  setBytesWrittenPerRep( 0 );
  setBytesModifyWrittenPerRep( 1*sizeof(Real_type) * mpa::D1D*mpa::D1D*mpa::D1D*m_NE ); // Y
  setBytesAtomicModifyWrittenPerRep( 0 );

  setFLOPsPerRep(m_NE * (2 * mpa::D1D * mpa::D1D * mpa::D1D * mpa::Q1D +
                         2 * mpa::D1D * mpa::D1D * mpa::Q1D * mpa::Q1D +
                         2 * mpa::D1D * mpa::Q1D * mpa::Q1D * mpa::Q1D + mpa::Q1D * mpa::Q1D * mpa::Q1D +
                         2 * mpa::Q1D * mpa::Q1D * mpa::Q1D * mpa::D1D +
                         2 * mpa::Q1D * mpa::Q1D * mpa::D1D * mpa::D1D +
                         2 * mpa::Q1D * mpa::D1D * mpa::D1D * mpa::D1D + mpa::D1D * mpa::D1D * mpa::D1D));
}

MASS3DPA::~MASS3DPA()
{
}

void MASS3DPA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_B,  mpa::Q1D*mpa::D1D, Real_type(1.0), vid);
  allocAndInitDataConst(m_Bt, mpa::Q1D*mpa::D1D, Real_type(1.0), vid);
  allocAndInitDataConst(m_D,  mpa::Q1D*mpa::Q1D*mpa::Q1D*m_NE, Real_type(1.0), vid);
  allocAndInitDataConst(m_X,  mpa::D1D*mpa::D1D*mpa::D1D*m_NE, Real_type(1.0), vid);
  allocAndInitDataConst(m_Y,  mpa::D1D*mpa::D1D*mpa::D1D*m_NE, Real_type(0.0), vid);
}

void MASS3DPA::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  addToChecksum(m_Y, mpa::D1D*mpa::D1D*mpa::D1D*m_NE, vid);
}

void MASS3DPA::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_B, vid);
  deallocData(m_Bt, vid);
  deallocData(m_D, vid);
  deallocData(m_X, vid);
  deallocData(m_Y, vid);
}

} // end namespace apps
} // end namespace rajaperf
