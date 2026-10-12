//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASSVEC3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace apps {

MASSVEC3DPA::MASSVEC3DPA(const RunParams &params)
    : KernelBase(rajaperf::Apps_MASSVEC3DPA, params)
{
  const Index_type NE_initial = 5208;
  setDefaultProblemSize(NE_initial * mvpa::DIM * mvpa::D1D * mvpa::D1D * mvpa::D1D);
  setDefaultReps(50);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(3);
  setProblemDimensionality(3);

  setUsesFeature(Launch);

  addVariantTunings();
}

void MASSVEC3DPA::setSize(Index_type target_size, Index_type target_reps)
{
  m_NE =
      std::max((target_size + (mvpa::DIM * mvpa::D1D * mvpa::D1D * mvpa::D1D) / 2) /
                   (mvpa::DIM * mvpa::D1D * mvpa::D1D * mvpa::D1D),
               Index_type(1));

  setActualProblemSize(m_NE * mvpa::DIM * mvpa::D1D * mvpa::D1D * mvpa::D1D);
  setRunReps( target_reps );

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesAllocatedPerRep( 1*sizeof(Real_type) * mvpa::Q1D*mvpa::D1D + // B
                           1*sizeof(Real_type) * mvpa::Q1D*mvpa::Q1D*mvpa::Q1D*m_NE + // D
                           2*sizeof(Real_type) * mvpa::D1D*mvpa::D1D*mvpa::D1D*mvpa::DIM*m_NE ); // X, Y
  setBytesReadPerRep( 1*sizeof(Real_type) * mvpa::Q1D*mvpa::D1D + // B
                      1*sizeof(Real_type) * mvpa::D1D*mvpa::D1D*mvpa::D1D*mvpa::DIM*m_NE + // X
                      1*sizeof(Real_type) * mvpa::Q1D*mvpa::Q1D*mvpa::Q1D*m_NE ); // D
  setBytesWrittenPerRep( 1*sizeof(Real_type) * mvpa::D1D*mvpa::D1D*mvpa::D1D*mvpa::DIM*m_NE ); // Y
  setBytesModifyWrittenPerRep( 0 );
  setBytesAtomicModifyWrittenPerRep(0);

  //3 for the dimension loop
  setFLOPsPerRep( m_NE*mvpa::DIM *
                  ( 2 * mvpa::D1D*mvpa::Q1D*mvpa::D1D*mvpa::D1D +
                    2 * mvpa::D1D*mvpa::Q1D*mvpa::Q1D*mvpa::D1D +
                    2 * mvpa::D1D*mvpa::Q1D*mvpa::Q1D*mvpa::Q1D +
                    mvpa::Q1D*mvpa::Q1D*mvpa::Q1D +
                    2 * mvpa::Q1D*mvpa::D1D*mvpa::Q1D*mvpa::Q1D +
                    2 * mvpa::Q1D*mvpa::D1D*mvpa::D1D*mvpa::Q1D +
                    2 * mvpa::Q1D*mvpa::D1D*mvpa::D1D*mvpa::D1D ) );
}

MASSVEC3DPA::~MASSVEC3DPA() {}

void MASSVEC3DPA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  allocAndInitDataConst(m_B, mvpa::Q1D*mvpa::D1D, 1.0, vid);
  allocAndInitDataConst(m_D, mvpa::Q1D*mvpa::Q1D*mvpa::Q1D*m_NE, 1.0, vid);
  allocAndInitDataConst(m_X, mvpa::D1D*mvpa::D1D*mvpa::D1D*mvpa::DIM*m_NE, 1.0, vid);
  allocAndInitDataConst(m_Y, mvpa::D1D*mvpa::D1D*mvpa::D1D*mvpa::DIM*m_NE, 0.0, vid);

}

void MASSVEC3DPA::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
   addToChecksum(m_Y, mvpa::DIM * mvpa::D1D * mvpa::D1D * mvpa::D1D * m_NE, vid);
}

void MASSVEC3DPA::tearDown(VariantID vid,
                           size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_B, vid);
  deallocData(m_D, vid);
  deallocData(m_X, vid);
  deallocData(m_Y, vid);
}

} // end namespace apps
} // end namespace rajaperf
