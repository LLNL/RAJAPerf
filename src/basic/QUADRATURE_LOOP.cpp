//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "QUADRATURE_LOOP.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{

QUADRATURE_LOOP::QUADRATURE_LOOP(const RunParams& params)
  : KernelBase(rajaperf::Basic_QUADRATURE_LOOP, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(2);
  setProblemDimensionality(2);

  setUsesFeature(Kernel);

  addVariantTunings();
}

void QUADRATURE_LOOP::setSize(Index_type target_size, Index_type target_reps)
{
  m_num_zones = (target_size + 26) / 27;

  setActualProblemSize(27 * m_num_zones);
  setRunReps(target_reps);

  setItsPerRep(27 * m_num_zones);
  setKernelsPerRep(1);

  setBytesAllocatedPerRep((27 * m_num_zones + 27 + 27 * m_num_zones) *
                          sizeof(Real_type));
  setBytesReadPerRep((27 * m_num_zones + 27 * m_num_zones) *
                     sizeof(Real_type));
  setBytesWrittenPerRep(27 * m_num_zones * sizeof(Real_type));
  setBytesModifyWrittenPerRep(0);
  setBytesAtomicModifyWrittenPerRep(0);
  setFLOPsPerRep(27 * m_num_zones);
}

QUADRATURE_LOOP::~QUADRATURE_LOOP()
{
}

void QUADRATURE_LOOP::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_values, 27 * m_num_zones, vid);
  allocAndInitData(m_weights, 27, vid);
  allocAndInitDataConst(m_output, 27 * m_num_zones, 0.0, vid);
}

void QUADRATURE_LOOP::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  addToChecksum(m_output, 27 * m_num_zones, vid);
}

void QUADRATURE_LOOP::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_values, vid);
  deallocData(m_weights, vid);
  deallocData(m_output, vid);
}

} // end namespace basic
} // end namespace rajaperf
