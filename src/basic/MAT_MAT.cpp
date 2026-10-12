//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf {
namespace basic {

MAT_MAT::MAT_MAT(const RunParams &params)
    : KernelBase(rajaperf::Basic_MAT_MAT, params)
{
  m_N_default = 1000;
  setDefaultProblemSize(m_N_default*m_N_default);
  setDefaultReps(5);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning); // Change to Inconsistent if internal reductions use atomics
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N_to_the_three_halves);

  setMaxPerfectLoopDimensions(2);
  setProblemDimensionality(2);

  setUsesFeature(Launch);

  addVariantTunings();
}

void MAT_MAT::setSize(Index_type target_size, Index_type target_reps)
{
  m_N = std::sqrt(target_size) + std::sqrt(2)-1;
  const Index_type num_tiles = RAJA_DIVIDE_CEILING_INT(m_N, MAT_MAT_TL_SZ);

  setActualProblemSize(m_N*m_N);
  setRunReps( target_reps );

  setItsPerRep( num_tiles*num_tiles * MAT_MAT_TL_SZ*MAT_MAT_TL_SZ );
  setKernelsPerRep(1);

  setBytesAllocatedPerRep( 3*sizeof(Real_type) * m_N*m_N ); // A, B, C
  setBytesReadPerRep( 2*sizeof(Real_type) * m_N*m_N ); // A, B
  setBytesWrittenPerRep( 1*sizeof(Real_type) * m_N*m_N  ); // C
  setBytesModifyWrittenPerRep( 0 );
  setBytesAtomicModifyWrittenPerRep( 0 );

  setFLOPsPerRep(2 * MAT_MAT_TL_SZ * MAT_MAT_TL_SZ * MAT_MAT_TL_SZ *
                 num_tiles * num_tiles * num_tiles);
}

MAT_MAT::~MAT_MAT() {}

void MAT_MAT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_A, m_N*m_N, 1.0, vid);
  allocAndInitDataConst(m_B, m_N*m_N, 1.0, vid);
  allocAndInitDataConst(m_C, m_N*m_N, 0.0, vid);
}

void MAT_MAT::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  addToChecksum(m_C, m_N*m_N, vid);
}

void MAT_MAT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_C, vid);
}

} // end namespace basic
} // end namespace rajaperf
