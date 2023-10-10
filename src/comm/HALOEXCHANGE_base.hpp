//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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

#ifndef RAJAPerf_Comm_HALOEXCHANGE_base_HPP
#define RAJAPerf_Comm_HALOEXCHANGE_base_HPP

#define HALOEXCHANGE_base_DATA_SETUP \
  std::vector<Real_ptr> vars = m_vars; \
  \
  Index_type num_neighbors = s_num_neighbors; \
  Index_type num_vars = m_num_vars; \
  std::vector<int> send_tags = m_send_tags; \
  std::vector<Int_ptr> pack_index_lists = m_pack_index_lists; \
  std::vector<Index_type> pack_index_list_lengths = m_pack_index_list_lengths; \
  std::vector<int> recv_tags = m_recv_tags; \
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

namespace comm
{

class HALOEXCHANGE_base : public KernelBase
{
public:

  HALOEXCHANGE_base(KernelID kid, const RunParams& params);

  ~HALOEXCHANGE_base();

  void setUp_base(const int my_mpi_rank, const int* mpi_dims,
             VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown_base(VariantID vid, size_t tune_idx);

protected:
  enum struct message_type : int
  {
    send,
    recv
  };

  struct Extent
  {
    Index_type i_min;
    Index_type i_max;
    Index_type j_min;
    Index_type j_max;
    Index_type k_min;
    Index_type k_max;
  };

  static const int s_num_neighbors = 26;
  static const int boundary_offsets[s_num_neighbors][3];

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

  std::vector<int> m_mpi_ranks;

  std::vector<int> m_send_tags;
  std::vector<Int_ptr> m_pack_index_lists;
  std::vector<Index_type > m_pack_index_list_lengths;

  std::vector<int> m_recv_tags;
  std::vector<Int_ptr> m_unpack_index_lists;
  std::vector<Index_type > m_unpack_index_list_lengths;

  Extent make_boundary_extent(
    const message_type msg_type,
    const int (&boundary_offset)[3],
    const Index_type halo_width, const Index_type* grid_dims);

  void create_lists(
      int my_mpi_rank,
      const int* mpi_dims,
      std::vector<int>& mpi_ranks,
      std::vector<int>& send_tags,
      std::vector<Int_ptr>& pack_index_lists,
      std::vector<Index_type >& pack_index_list_lengths,
      std::vector<int>& recv_tags,
      std::vector<Int_ptr>& unpack_index_lists,
      std::vector<Index_type >& unpack_index_list_lengths,
      const Index_type halo_width, const Index_type* grid_dims,
      const Index_type num_neighbors,
      VariantID vid);

  void destroy_lists(
      std::vector<Int_ptr>& pack_index_lists,
      std::vector<Int_ptr>& unpack_index_lists,
      const Index_type num_neighbors,
      VariantID vid);
};

} // end namespace comm
} // end namespace rajaperf

#endif // closing endif for header file include guard