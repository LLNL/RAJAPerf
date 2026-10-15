//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POINTER_CHASE kernel reference implementation:
///
/// Index_type idx = start_idx;
/// for (std::uint64_t step = 0; step < num_steps; ++step) {
///   idx = next[idx];
/// }
/// result[0] = idx;
///

#ifndef RAJAPerf_Basic_POINTER_CHASE_HPP
#define RAJAPerf_Basic_POINTER_CHASE_HPP

#include "common/KernelBase.hpp"

#include <cstdint>

#define POINTER_CHASE_DATA_SETUP                                              \
  Index_ptr next = m_next;                                                    \
  Index_ptr result = m_result;                                                \
  const Index_type start_idx = m_start_idx;                                   \
  const std::uint64_t num_steps = m_num_steps;

#define POINTER_CHASE_BODY                                                    \
  Index_type idx = start_idx;                                                 \
  for (std::uint64_t step = 0; step < num_steps; ++step) {                    \
    idx = next[idx];                                                          \
  }                                                                           \
  result[0] = idx;

namespace rajaperf
{
class RunParams;

namespace basic
{

class POINTER_CHASE : public KernelBase
{
public:
  POINTER_CHASE(const RunParams& params);
  ~POINTER_CHASE();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();

  void runSeqVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);

private:
  Index_ptr m_next;
  Index_ptr m_result;
  Index_type m_N;
  Index_type m_start_idx;
  std::uint64_t m_traversals;
  std::uint64_t m_num_steps;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
