//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POINTER_CHASE.hpp"

#include "common/DataUtils.hpp"
#include "common/RunParams.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace rajaperf
{
namespace basic
{

POINTER_CHASE::POINTER_CHASE(const RunParams& params)
  : KernelBase(rajaperf::Basic_POINTER_CHASE, params),
    m_traversals(params.getPointerChaseTraversals())
{
  setDefaultProblemSize(1000000);
  setDefaultReps(10);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::Consistent);
  setChecksumTolerance(ChecksumTolerance::zero);

  setComplexity(Complexity::N);
  setMaxPerfectLoopDimensions(1);
  setProblemDimensionality(1);
  setProblemSizeAlignment(ProblemSizeAlignment::OneDimensional);

  setUsesFeature(Forall);

  addVariantTunings();
}

void POINTER_CHASE::setSize(Index_type target_size, Index_type target_reps)
{
  m_N = std::max<Index_type>(target_size, 1);

  const std::uint64_t n = static_cast<std::uint64_t>(m_N);
  const std::uint64_t metadata_max =
      static_cast<std::uint64_t>(std::numeric_limits<Index_type>::max());

  if (m_traversals > std::numeric_limits<std::uint64_t>::max() / n) {
    throw std::overflow_error("POINTER_CHASE num_steps overflow");
  }

  m_num_steps = n * m_traversals;

  if (m_num_steps > metadata_max ||
      m_num_steps > metadata_max / sizeof(Index_type) ||
      n > metadata_max / sizeof(Index_type) - 1) {
    throw std::overflow_error("POINTER_CHASE metadata overflow");
  }

  setActualProblemSize(m_N);
  setRunReps(target_reps);

  setItsPerRep(static_cast<Index_type>(m_num_steps));
  setKernelsPerRep(1);

  setBytesAllocatedPerRep(
      static_cast<Index_type>((n + 1) * sizeof(Index_type)));
  setBytesReadPerRep(
      static_cast<Index_type>(m_num_steps * sizeof(Index_type)));
  setBytesWrittenPerRep(sizeof(Index_type));
  setBytesModifyWrittenPerRep(0);
  setBytesAtomicModifyWrittenPerRep(0);
  setFLOPsPerRep(0);
}

POINTER_CHASE::~POINTER_CHASE()
{
}

void POINTER_CHASE::setUp(VariantID vid,
                          size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  auto next_data = allocDataForInit(m_next, m_N, vid);
  auto result_data = allocAndInitDataConstForInit(
      m_result, 1, Index_type(-1), vid);

  std::vector<Index_type> permutation(static_cast<size_t>(m_N));
  std::iota(permutation.begin(), permutation.end(), Index_type(0));
  std::mt19937 generator(12345);
  std::shuffle(permutation.begin(), permutation.end(), generator);

  for (Index_type i = 0; i < m_N; ++i) {
    m_next[permutation[static_cast<size_t>(i)]] =
        permutation[static_cast<size_t>((i + 1) % m_N)];
  }

  m_start_idx = permutation[0];
  if (m_start_idx == 0 && m_N > 1) {
    m_start_idx = permutation[1];
  }
}

void POINTER_CHASE::updateChecksum(
    VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  auto result_data = scopedMoveData(m_result, 1, vid);

  if (m_result[0] != m_start_idx) {
    throw std::runtime_error("POINTER_CHASE result does not match start index");
  }

  addToChecksum(m_result[0]);
}

void POINTER_CHASE::tearDown(VariantID vid,
                             size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_next, vid);
  deallocData(m_result, vid);
}

} // end namespace basic
} // end namespace rajaperf
