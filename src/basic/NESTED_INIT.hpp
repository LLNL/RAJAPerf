//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NESTED_INIT kernel reference implementation:
///
/// for (Index_type k = 0; k < nk; ++k ) {
///   for (Index_type j = 0; j < nj; ++j ) {
///     for (Index_type i = 0; i < ni; ++i ) {
///       array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
///     }
///   }
/// }
///

#ifndef RAJAPerf_Basic_NESTED_INIT_HPP
#define RAJAPerf_Basic_NESTED_INIT_HPP


#define NESTED_INIT_DATA_SETUP \
  Real_ptr array = m_array; \
  Index_type ni = m_ni; \
  Index_type nj = m_nj; \
  Index_type nk = m_nk;

#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class NESTED_INIT : public KernelBase
{
public:

  NESTED_INIT(const RunParams& params);

  ~NESTED_INIT();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineKokkosVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runKokkosVariant(VariantID vid);

  void runSeqVariantFornest(VariantID vid);
  template < size_t tile_k, size_t tile_j, size_t tile_i >
  void runSeqVariantFornestRuntimeTiled(VariantID vid);
  void runSeqVariantFornestAutoTiled(VariantID vid);
  void runOpenMPVariantFornest(VariantID vid);
  template < size_t tile_k, size_t tile_j, size_t tile_i >
  void runOpenMPVariantFornestRuntimeTiled(VariantID vid);
  void runOpenMPVariantFornestAutoTiled(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runCudaVariantFornest(VariantID vid);
  template < size_t block_size, size_t tile_k, size_t tile_j, size_t tile_i >
  void runCudaVariantFornestRuntimeTiled(VariantID vid);
  template < size_t block_size >
  void runCudaVariantFornestAutoTiled(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantFornest(VariantID vid);
  template < size_t block_size, size_t tile_k, size_t tile_j, size_t tile_i >
  void runHipVariantFornestRuntimeTiled(VariantID vid);
  template < size_t block_size >
  void runHipVariantFornestAutoTiled(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantFornest(VariantID vid);
  template < size_t work_group_size, size_t tile_k, size_t tile_j, size_t tile_i >
  void runSyclVariantFornestRuntimeTiled(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantFornestAutoTiled(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  Index_type m_array_length;

  Real_ptr m_array;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_n_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
