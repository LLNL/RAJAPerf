//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DYANAMIC_TILE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <cmath>

namespace rajaperf
{
namespace basic
{

DYANAMIC_TILE::DYANAMIC_TILE(const RunParams& params)
  : KernelBase(rajaperf::Basic_DYANAMIC_TILE, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::normal);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(3);
  setProblemDimensionality(3);

  setUsesFeature(Kernel);

  addVariantTunings();
}

void DYANAMIC_TILE::setSize(Index_type target_size, Index_type target_reps)
{
  const Index_type target_per_kernel = (target_size + 2) / 3;
  Index_type base =
    static_cast<Index_type>(std::cbrt(static_cast<double>(target_per_kernel)));
  if (base < 1) {
    base = 1;
  }

  auto ceil_div = [](Index_type len, Index_type div) {
                    return (len + div - 1) / div;
                  };

  m_ni0 = 8 * base;
  m_nj0 = 2 * base;
  m_nk0 = ceil_div(target_per_kernel, m_ni0 * m_nj0);

  m_ni1 = 2 * base;
  m_nj1 = 8 * base;
  m_nk1 = ceil_div(target_per_kernel, m_ni1 * m_nj1);

  m_ni2 = 4 * base;
  m_nj2 = 4 * base;
  m_nk2 = ceil_div(target_per_kernel, m_ni2 * m_nj2);

  m_len0 = m_ni0 * m_nj0 * m_nk0;
  m_len1 = m_ni1 * m_nj1 * m_nk1;
  m_len2 = m_ni2 * m_nj2 * m_nk2;

  m_offset0 = 0;
  m_offset1 = m_len0;
  m_offset2 = m_len0 + m_len1;

  setActualProblemSize(m_len0 + m_len1 + m_len2);
  setRunReps(target_reps);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(3);

  setBytesAllocatedPerRep(2 * sizeof(Real_type) * getActualProblemSize());
  setBytesReadPerRep(1 * sizeof(Real_type) * getActualProblemSize());
  setBytesWrittenPerRep(1 * sizeof(Real_type) * getActualProblemSize());
  setBytesModifyWrittenPerRep(0);
  setBytesAtomicModifyWrittenPerRep(0);
  setFLOPsPerRep(4 * getActualProblemSize());
}

DYANAMIC_TILE::~DYANAMIC_TILE()
{
}

void DYANAMIC_TILE::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_input, getActualProblemSize(), vid);
  allocAndInitDataConst(m_output, getActualProblemSize(), 0.0, vid);
}

void DYANAMIC_TILE::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  addToChecksum(m_output, getActualProblemSize(), vid);
}

void DYANAMIC_TILE::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_input, vid);
  deallocData(m_output, vid);
}

} // end namespace basic
} // end namespace rajaperf
