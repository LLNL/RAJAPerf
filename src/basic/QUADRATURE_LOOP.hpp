//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// QUADRATURE_LOOP kernel reference implementation:
///
/// for (Index_type zone = 0; zone < num_zones; ++zone ) {
///   for (Index_type q = 0; q < 27; ++q ) {
///     output[27 * zone + q] = weights[q] * values[27 * zone + q];
///   }
/// }
///

#ifndef RAJAPerf_Basic_QUADRATURE_LOOP_HPP
#define RAJAPerf_Basic_QUADRATURE_LOOP_HPP


#define QUADRATURE_LOOP_DATA_SETUP \
  Real_ptr values = m_values; \
  Real_ptr weights = m_weights; \
  Real_ptr output = m_output; \
  Index_type num_zones = m_num_zones;

#define QUADRATURE_LOOP_BODY  \
  output[27 * zone + q] = weights[q] * values[27 * zone + q];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class QUADRATURE_LOOP : public KernelBase
{
public:

  QUADRATURE_LOOP(const RunParams& params);

  ~QUADRATURE_LOOP();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_num_zones;

  Real_ptr m_values;
  Real_ptr m_weights;
  Real_ptr m_output;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
