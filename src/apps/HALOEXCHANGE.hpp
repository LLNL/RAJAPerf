//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALOEXCHANGE kernel reference implementation:
///
/// // pack message for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Real_ptr buffer = buffers[l];
///   Int_ptr list = pack_index_lists[l];
///   Index_type  len  = pack_index_list_lengths[l];
///   // pack part of each variable
///   for (Index_type v = 0; v < num_vars; ++v) {
///     Real_ptr var = vars[v];
///     for (Index_type i = 0; i < len; i++) {
///       HALOEXCHANGE_PACK_BODY;
///     }
///     buffer += len;
///   }
///   // send message to neighbor
/// }
///
/// // unpack messages for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   // receive message from neighbor
///   Real_ptr buffer = buffers[l];
///   Int_ptr list = unpack_index_lists[l];
///   Index_type  len  = unpack_index_list_lengths[l];
///   // unpack part of each variable
///   for (Index_type v = 0; v < num_vars; ++v) {
///     Real_ptr var = vars[v];
///     for (Index_type i = 0; i < len; i++) {
///       HALOEXCHANGE_UNPACK_BODY;
///     }
///     buffer += len;
///   }
/// }
///

#ifndef RAJAPerf_Apps_HALOEXCHANGE_HPP
#define RAJAPerf_Apps_HALOEXCHANGE_HPP

#define HALOEXCHANGE_DATA_SETUP \
  std::vector<Real_ptr> vars = m_vars; \
  std::vector<Real_ptr> buffers = m_buffers; \
\
  Index_type num_neighbors = s_num_neighbors; \
  Index_type num_vars = m_num_vars; \
  std::vector<Int_ptr> pack_index_lists = m_pack_index_lists; \
  std::vector<Index_type> pack_index_list_lengths = m_pack_index_list_lengths; \
  std::vector<Int_ptr> unpack_index_lists = m_unpack_index_lists; \
  std::vector<Index_type> unpack_index_list_lengths = m_unpack_index_list_lengths;

#define HALOEXCHANGE_PACK_BODY \
  buffer[i] = var[list[i]];

#define HALOEXCHANGE_UNPACK_BODY \
  var[list[i]] = buffer[i];


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
class RunParams;

namespace apps
{

class HALOEXCHANGE : public KernelBase
{
public:

  HALOEXCHANGE(const RunParams& params);

  ~HALOEXCHANGE();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  static const int s_num_neighbors = 26;

  Index_type m_grid_dims[3];
  Index_type m_halo_width;
  Index_type m_num_vars;

  Index_type m_grid_dims_default[3];
  Index_type m_halo_width_default;
  Index_type m_num_vars_default;

  Index_type m_grid_plus_halo_dims[3];
  Index_type m_var_size;
  Index_type m_var_halo_size;

  std::vector<Real_ptr> m_vars;
  std::vector<Real_ptr> m_buffers;

  std::vector<Int_ptr> m_pack_index_lists;
  std::vector<Index_type > m_pack_index_list_lengths;
  std::vector<Int_ptr> m_unpack_index_lists;
  std::vector<Index_type > m_unpack_index_list_lengths;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
