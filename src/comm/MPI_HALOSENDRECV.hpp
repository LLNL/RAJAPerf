//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// MPI_HALOSENDRECV kernel reference implementation:
///
/// // post a recv for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Index_type len = unpack_index_list_lengths[l];
///   MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
///       mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
/// }
///
/// // pack a buffer for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Real_ptr buffer = pack_buffers[l];
///   Int_ptr list = pack_index_lists[l];
///   Index_type  len  = pack_index_list_lengths[l];
///   // pack part of each variable
///   for (Index_type v = 0; v < num_vars; ++v) {
///     Real_ptr var = vars[v];
///     for (Index_type i = 0; i < len; i++) {
///       buffer[i] = var[list[i]];
///     }
///     buffer += len;
///   }
///   // send buffer to neighbor
///   MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
///       mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
/// }
///
/// // unpack a buffer for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   // receive buffer from neighbor
///   MPI_Wait(&unpack_mpi_requests[l], MPI_STATUS_IGNORE);
///   Real_ptr buffer = unpack_buffers[l];
///   Int_ptr list = unpack_index_lists[l];
///   Index_type  len  = unpack_index_list_lengths[l];
///   // unpack part of each variable
///   for (Index_type v = 0; v < num_vars; ++v) {
///     Real_ptr var = vars[v];
///     for (Index_type i = 0; i < len; i++) {
///       var[list[i]] = buffer[i];
///     }
///     buffer += len;
///   }
/// }
///
/// // wait for all sends to complete
/// MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);
///


#ifndef RAJAPerf_Comm_MPI_HALOSENDRECV_HPP
#define RAJAPerf_Comm_MPI_HALOSENDRECV_HPP

#define MPI_HALOSENDRECV_DATA_SETUP \
  HALO_BASE_DATA_SETUP \
  \
  Index_type num_vars = m_num_vars; \
  \
  std::vector<int> mpi_ranks = m_mpi_ranks; \
  \
  std::vector<MPI_Request> pack_mpi_requests(num_neighbors); \
  std::vector<MPI_Request> unpack_mpi_requests(num_neighbors); \
  \
  std::vector<Real_ptr> send_buffers = m_send_buffers; \
  std::vector<Real_ptr> recv_buffers = m_recv_buffers;


#include "HALO_base.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#include <vector>
#include <array>

namespace rajaperf
{
namespace comm
{

class MPI_HALOSENDRECV : public HALO_base
{
public:

  MPI_HALOSENDRECV(const RunParams& params);

  ~MPI_HALOSENDRECV();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

private:
  int m_mpi_size = -1;
  int m_my_mpi_rank = -1;
  std::array<int, 3> m_mpi_dims = {-1, -1, -1};

  Index_type m_num_vars;
  Index_type m_var_size;

  std::vector<Real_ptr> m_send_buffers;
  std::vector<Real_ptr> m_recv_buffers;
};

} // end namespace comm
} // end namespace rajaperf

#endif
#endif // closing endif for header file include guard
