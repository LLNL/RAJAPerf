//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALO_base provides a common starting point for the other HALO_ classes.
///

#ifndef RAJAPerf_Comm_HALO_BASE_HPP
#define RAJAPerf_Comm_HALO_BASE_HPP

#define HALO_BASE_DATA_SETUP \
  Index_type num_neighbors = s_num_neighbors; \
  std::vector<int> send_tags = m_send_tags; \
  std::vector<Int_ptr> pack_index_lists = m_pack_index_lists; \
  std::vector<Index_type> pack_index_list_lengths = m_pack_index_list_lengths; \
  std::vector<int> recv_tags = m_recv_tags; \
  std::vector<Int_ptr> unpack_index_lists = m_unpack_index_lists; \
  std::vector<Index_type> unpack_index_list_lengths = m_unpack_index_list_lengths;

#define HALO_PACK_BODY \
  buffer[i] = var[list[i]];

#define HALO_UNPACK_BODY \
  var[list[i]] = buffer[i];


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
class RunParams;

struct direct_dispatch_helper
{
  template < typename... Ts >
  using dispatch_policy = RAJA::direct_dispatch<Ts...>;
  static std::string get_name() { return "direct"; }
};

struct indirect_function_call_dispatch_helper
{
  template < typename... Ts >
  using dispatch_policy = RAJA::indirect_function_call_dispatch;
  static std::string get_name() { return "funcptr"; }
};

struct indirect_virtual_function_dispatch_helper
{
  template < typename... Ts >
  using dispatch_policy = RAJA::indirect_virtual_function_dispatch;
  static std::string get_name() { return "virtfunc"; }
};

using workgroup_dispatch_helpers = camp::list<
    direct_dispatch_helper,
    indirect_function_call_dispatch_helper,
    indirect_virtual_function_dispatch_helper >;

using hip_workgroup_dispatch_helpers = camp::list<
    direct_dispatch_helper
#ifdef RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL
   ,indirect_function_call_dispatch_helper
   ,indirect_virtual_function_dispatch_helper
#endif
    >;

namespace comm
{

class HALO_base : public KernelBase
{
public:

  HALO_base(KernelID kid, const RunParams& params);

  ~HALO_base();

  void setUp_base(const int my_mpi_rank, const int* mpi_dims,
             VariantID vid, size_t tune_idx);
  void tearDown_base(VariantID vid, size_t tune_idx);

  struct Packer {
    Real_ptr buffer;
    Real_ptr var;
    Int_ptr list;
    RAJA_HOST_DEVICE void operator()(Index_type i) const {
      HALO_PACK_BODY;
    }
  };

  struct UnPacker {
    Real_ptr buffer;
    Real_ptr var;
    Int_ptr list;
    RAJA_HOST_DEVICE void operator()(Index_type i) const {
      HALO_UNPACK_BODY;
    }
  };

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
  static const int s_boundary_offsets[s_num_neighbors][3];

  static Index_type s_grid_dims_default[3];
  static Index_type s_halo_width_default;
  static Index_type s_num_vars_default;

  Index_type m_grid_dims[3];
  Index_type m_halo_width;

  Index_type m_grid_plus_halo_dims[3];
  Index_type m_grid_plus_halo_size;

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
