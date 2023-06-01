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

#ifndef RAJAPerf_Apps_HALOEXCHANGE_base_HPP
#define RAJAPerf_Apps_HALOEXCHANGE_base_HPP

#define HALOEXCHANGE_base_DATA_SETUP \
  std::vector<Real_ptr> vars = m_vars; \
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

class HALOEXCHANGE_base : public KernelBase
{
public:

  HALOEXCHANGE_base(KernelID kid, const RunParams& params);

  ~HALOEXCHANGE_base();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

protected:
  enum struct location : int
  {
    low_phony,
    low_interior,
    all_interior,
    high_interior,
    high_phony
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

  std::vector<Int_ptr> m_pack_index_lists;
  std::vector<Index_type > m_pack_index_list_lengths;
  std::vector<Int_ptr> m_unpack_index_lists;
  std::vector<Index_type > m_unpack_index_list_lengths;

  Extent make_extent(
    location x_extent, location y_extent, location z_extent,
    const Index_type halo_width, const Index_type* grid_dims);

  void create_pack_lists(
      std::vector<Int_ptr>& pack_index_lists,
      std::vector<Index_type >& pack_index_list_lengths,
      const Index_type halo_width, const Index_type* grid_dims,
      const Index_type num_neighbors,
      VariantID vid);

  void destroy_pack_lists(
      std::vector<Int_ptr>& pack_index_lists,
      const Index_type num_neighbors,
      VariantID vid);

  void create_unpack_lists(
      std::vector<Int_ptr>& unpack_index_lists,
      std::vector<Index_type >& unpack_index_list_lengths,
      const Index_type halo_width, const Index_type* grid_dims,
      const Index_type num_neighbors,
      VariantID vid);

  void destroy_unpack_lists(
      std::vector<Int_ptr>& unpack_index_lists,
      const Index_type num_neighbors,
      VariantID vid);

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  void create_rank_list(
      int my_mpi_rank, int mpi_size,
      std::vector<int>& mpi_ranks,
      const Index_type num_neighbors,
      VariantID vid);

  void destroy_rank_list(
      const Index_type num_neighbors,
      VariantID vid);
#endif
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
