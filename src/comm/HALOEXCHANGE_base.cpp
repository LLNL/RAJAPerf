//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE_base.hpp"

#include "RAJA/RAJA.hpp"

#include <utility>
#include <cmath>
#include <map>

namespace rajaperf
{
namespace comm
{

HALOEXCHANGE_base::HALOEXCHANGE_base(KernelID kid, const RunParams& params)
  : KernelBase(kid, params)
{
  m_grid_dims_default[0] = 100;
  m_grid_dims_default[1] = 100;
  m_grid_dims_default[2] = 100;
  m_halo_width_default   = 1;
  m_num_vars_default     = 3;

  setDefaultProblemSize( m_grid_dims_default[0] *
                         m_grid_dims_default[1] *
                         m_grid_dims_default[2] );

  double cbrt_run_size = std::cbrt(getTargetProblemSize());

  m_grid_dims[0] = cbrt_run_size;
  m_grid_dims[1] = cbrt_run_size;
  m_grid_dims[2] = cbrt_run_size;
  m_halo_width = m_halo_width_default;
  m_num_vars   = m_num_vars_default;

  m_grid_plus_halo_dims[0] = m_grid_dims[0] + 2*m_halo_width;
  m_grid_plus_halo_dims[1] = m_grid_dims[1] + 2*m_halo_width;
  m_grid_plus_halo_dims[2] = m_grid_dims[2] + 2*m_halo_width;
  m_var_size = m_grid_plus_halo_dims[0] *
               m_grid_plus_halo_dims[1] *
               m_grid_plus_halo_dims[2] ;

  setActualProblemSize( m_grid_dims[0] * m_grid_dims[1] * m_grid_dims[1] );

  setItsPerRep( m_num_vars * (m_var_size - getActualProblemSize()) );
}

HALOEXCHANGE_base::~HALOEXCHANGE_base()
{
}

void HALOEXCHANGE_base::setUp_base(const int my_mpi_rank, const int* mpi_dims,
                                   VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  m_vars.resize(m_num_vars, nullptr);
  for (Index_type v = 0; v < m_num_vars; ++v) {
    allocAndInitData(m_vars[v], m_var_size, vid);
    auto reset_var = scopedMoveData(m_vars[v], m_var_size, vid);

    Real_ptr var = m_vars[v];

    for (Index_type i = 0; i < m_var_size; i++) {
      var[i] = i + v;
    }
  }

  m_mpi_ranks.resize(s_num_neighbors, -1);
  m_send_tags.resize(s_num_neighbors, -1);
  m_pack_index_lists.resize(s_num_neighbors, nullptr);
  m_pack_index_list_lengths.resize(s_num_neighbors, 0);
  m_recv_tags.resize(s_num_neighbors, -1);
  m_unpack_index_lists.resize(s_num_neighbors, nullptr);
  m_unpack_index_list_lengths.resize(s_num_neighbors, 0);
  create_lists(my_mpi_rank, mpi_dims, m_mpi_ranks,
      m_send_tags, m_pack_index_lists, m_pack_index_list_lengths,
      m_recv_tags, m_unpack_index_lists, m_unpack_index_list_lengths,
      m_halo_width, m_grid_dims,
      s_num_neighbors, vid);
}

void HALOEXCHANGE_base::updateChecksum(VariantID vid, size_t tune_idx)
{
  for (Real_ptr var : m_vars) {
    checksum[vid][tune_idx] += calcChecksum(var, m_var_size, vid);
  }
}

void HALOEXCHANGE_base::tearDown_base(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  destroy_lists(m_pack_index_lists, m_unpack_index_lists, s_num_neighbors, vid);
  m_unpack_index_list_lengths.clear();
  m_unpack_index_lists.clear();
  m_recv_tags.clear();
  m_pack_index_list_lengths.clear();
  m_pack_index_lists.clear();
  m_send_tags.clear();
  m_mpi_ranks.clear();

  for (int v = 0; v < m_num_vars; ++v) {
    deallocData(m_vars[v], vid);
  }
  m_vars.clear();
}


const int HALOEXCHANGE_base::boundary_offsets[HALOEXCHANGE_base::s_num_neighbors][3]{

  // faces
  {-1,  0,  0},
  { 1,  0,  0},
  { 0, -1,  0},
  { 0,  1,  0},
  { 0,  0, -1},
  { 0,  0,  1},

  // edges
  {-1, -1,  0},
  {-1,  1,  0},
  { 1, -1,  0},
  { 1,  1,  0},
  {-1,  0, -1},
  {-1,  0,  1},
  { 1,  0, -1},
  { 1,  0,  1},
  { 0, -1, -1},
  { 0, -1,  1},
  { 0,  1, -1},
  { 0,  1,  1},

  // corners
  {-1, -1, -1},
  {-1, -1,  1},
  {-1,  1, -1},
  {-1,  1,  1},
  { 1, -1, -1},
  { 1, -1,  1},
  { 1,  1, -1},
  { 1,  1,  1}

};

HALOEXCHANGE_base::Extent HALOEXCHANGE_base::make_boundary_extent(
    const HALOEXCHANGE_base::message_type msg_type,
    const int (&boundary_offset)[3],
    const Index_type halo_width, const Index_type* grid_dims)
{
  if (msg_type != message_type::send &&
      msg_type != message_type::recv) {
    throw std::runtime_error("make_boundary_extent: Invalid message type");
  }
  auto get_bounds = [&](int offset, Index_type dim_size) {
    std::pair<Index_type, Index_type> bounds;
    switch (offset) {
    case -1:
      if (msg_type == message_type::send) {
        bounds.first  = halo_width;
        bounds.second = halo_width + halo_width;
      } else { // (msg_type == message_type::recv)
        bounds.first  = 0;
        bounds.second = halo_width;
      }
      break;
    case 0:
      bounds.first  = halo_width;
      bounds.second = halo_width + dim_size;
      break;
    case 1:
      if (msg_type == message_type::send) {
        bounds.first  = halo_width + dim_size - halo_width;
        bounds.second = halo_width + dim_size;
      } else { // (msg_type == message_type::recv)
        bounds.first  = halo_width + dim_size;
        bounds.second = halo_width + dim_size + halo_width;
      }
      break;
    default:
      throw std::runtime_error("make_extent: Invalid location");
    }
    return bounds;
  };
  auto x_bounds = get_bounds(boundary_offset[0], grid_dims[0]);
  auto y_bounds = get_bounds(boundary_offset[1], grid_dims[1]);
  auto z_bounds = get_bounds(boundary_offset[2], grid_dims[2]);
  return {x_bounds.first, x_bounds.second,
          y_bounds.first, y_bounds.second,
          z_bounds.first, z_bounds.second};
}


//
// Function to generate mpi decomposition and index lists for packing and unpacking.
//
void HALOEXCHANGE_base::create_lists(
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
    VariantID vid)
{
  int my_mpi_idx[3]{-1,-1,-1};
  my_mpi_idx[2] = my_mpi_rank / (mpi_dims[0]*mpi_dims[1]);
  my_mpi_idx[1] = (my_mpi_rank - my_mpi_idx[2]*(mpi_dims[0]*mpi_dims[1])) / mpi_dims[0];
  my_mpi_idx[0] = my_mpi_rank - my_mpi_idx[2]*(mpi_dims[0]*mpi_dims[1]) - my_mpi_idx[1]*mpi_dims[0];

  auto get_boundary_idx = [&](const int (&boundary_offset)[3]) {
    return (boundary_offset[0]+1) + 3*(boundary_offset[1]+1) + 9*(boundary_offset[2]+1);
  };

  std::map<int, int> boundary_idx_to_tag;
  for (Index_type l = 0; l < num_neighbors; ++l) {
    boundary_idx_to_tag[get_boundary_idx(boundary_offsets[l])] = l;
  }

  const Index_type grid_i_stride = 1;
  const Index_type grid_j_stride = grid_dims[0] + 2*halo_width;
  const Index_type grid_k_stride = grid_j_stride * (grid_dims[1] + 2*halo_width);

  for (Index_type l = 0; l < num_neighbors; ++l) {

    const int (&boundary_offset)[3] = boundary_offsets[l];

    int neighbor_boundary_offset[3]{-1, -1, -1};
    for (int dim = 0; dim < 3; ++dim) {
      neighbor_boundary_offset[dim] = -boundary_offset[dim];
    }

    int neighbor_mpi_idx[3] = {my_mpi_idx[0]+boundary_offset[0],
                               my_mpi_idx[1]+boundary_offset[1],
                               my_mpi_idx[2]+boundary_offset[2]};

    // fix neighbor mpi index on periodic boundaries
    for (int dim = 0; dim < 3; ++dim) {
      if (neighbor_mpi_idx[dim] >= mpi_dims[dim]) {
        neighbor_mpi_idx[dim] = 0;
      } else if (neighbor_mpi_idx[dim] < 0) {
        neighbor_mpi_idx[dim] = mpi_dims[dim]-1;
      }
    }

    mpi_ranks[l] = neighbor_mpi_idx[0] + mpi_dims[0]*(neighbor_mpi_idx[1] + mpi_dims[1]*neighbor_mpi_idx[2]);

    {
      // pack and send
      send_tags[l] = boundary_idx_to_tag[get_boundary_idx(boundary_offset)];
      Extent extent = make_boundary_extent(message_type::send,
                                           boundary_offset,
                                           halo_width, grid_dims);

      pack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                   (extent.j_max - extent.j_min) *
                                   (extent.k_max - extent.k_min) ;

      allocAndInitData(pack_index_lists[l], pack_index_list_lengths[l], vid);
      auto reset_list = scopedMoveData(pack_index_lists[l], pack_index_list_lengths[l], vid);

      Int_ptr pack_list = pack_index_lists[l];

      Index_type list_idx = 0;
      for (Index_type kk = extent.k_min; kk < extent.k_max; ++kk) {
        for (Index_type jj = extent.j_min; jj < extent.j_max; ++jj) {
          for (Index_type ii = extent.i_min; ii < extent.i_max; ++ii) {

            Index_type pack_idx = ii * grid_i_stride +
                                  jj * grid_j_stride +
                                  kk * grid_k_stride ;

            pack_list[list_idx] = pack_idx;

            list_idx += 1;
          }
        }
      }
    }

    {
      // receive and unpack
      recv_tags[l] = boundary_idx_to_tag[get_boundary_idx(neighbor_boundary_offset)];
      Extent extent = make_boundary_extent(message_type::recv,
                                           boundary_offset,
                                           halo_width, grid_dims);

      unpack_index_list_lengths[l] = (extent.i_max - extent.i_min) *
                                     (extent.j_max - extent.j_min) *
                                     (extent.k_max - extent.k_min) ;

      allocAndInitData(unpack_index_lists[l], unpack_index_list_lengths[l], vid);
      auto reset_list = scopedMoveData(unpack_index_lists[l], unpack_index_list_lengths[l], vid);

      Int_ptr unpack_list = unpack_index_lists[l];

      Index_type list_idx = 0;
      for (Index_type kk = extent.k_min; kk < extent.k_max; ++kk) {
        for (Index_type jj = extent.j_min; jj < extent.j_max; ++jj) {
          for (Index_type ii = extent.i_min; ii < extent.i_max; ++ii) {

            Index_type unpack_idx = ii * grid_i_stride +
                                    jj * grid_j_stride +
                                    kk * grid_k_stride ;

            unpack_list[list_idx] = unpack_idx;

            list_idx += 1;
          }
        }
      }
    }
  }
}

//
// Function to destroy packing and unpacking index lists.
//
void HALOEXCHANGE_base::destroy_lists(
    std::vector<Int_ptr>& pack_index_lists,
    std::vector<Int_ptr>& unpack_index_lists,
    const Index_type num_neighbors,
    VariantID vid)
{
  for (Index_type l = 0; l < num_neighbors; ++l) {
    deallocData(pack_index_lists[l], vid);
  }
  for (Index_type l = 0; l < num_neighbors; ++l) {
    deallocData(unpack_index_lists[l], vid);
  }
}

} // end namespace comm
} // end namespace rajaperf
