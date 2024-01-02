//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALO_EXCHANGE_FUSED kernel reference implementation:
///
/// // post a recv for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Index_type len = unpack_index_list_lengths[l];
///   MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
///       mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
/// }
///
/// // pack buffers for neighbors
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Real_ptr buffer = pack_buffers[l];
///   Int_ptr list = pack_index_lists[l];
///   Index_type len = pack_index_list_lengths[l];
///   // pack part of each variable
///   for (Index_type v = 0; v < num_vars; ++v) {
///     Real_ptr var = vars[v];
///     for (Index_type i = 0; i < len; i++) {
///       buffer[i] = var[list[i]];
///     }
///     buffer += len;
///   }
/// }
///
/// // send buffers to neighbors
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
///       mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
/// }
///
/// // wait for all recvs to complete
/// MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);
///
/// // unpack buffers for neighbors
/// for (Index_type l = 0; l < num_neighbors; ++l) {
///   Real_ptr buffer = unpack_buffers[l];
///   Int_ptr list = unpack_index_lists[l];
///   Index_type len = unpack_index_list_lengths[l];
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

#ifndef RAJAPerf_Comm_HALO_EXCHANGE_FUSED_HPP
#define RAJAPerf_Comm_HALO_EXCHANGE_FUSED_HPP

#define HALO_EXCHANGE_FUSED_DATA_SETUP \
  HALO_BASE_DATA_SETUP \
  \
  Index_type num_vars = m_num_vars; \
  std::vector<Real_ptr> vars = m_vars; \
  \
  std::vector<int> mpi_ranks = m_mpi_ranks; \
  \
  std::vector<MPI_Request> pack_mpi_requests(num_neighbors); \
  std::vector<MPI_Request> unpack_mpi_requests(num_neighbors); \
  \
  const DataSpace dataSpace = getDataSpace(vid); \
  \
  const bool separate_buffers = (getMPIDataSpace(vid) == DataSpace::Copy); \
  \
  std::vector<Real_ptr> pack_buffers = m_pack_buffers; \
  std::vector<Real_ptr> unpack_buffers = m_unpack_buffers; \
  \
  std::vector<Real_ptr> send_buffers = m_send_buffers; \
  std::vector<Real_ptr> recv_buffers = m_recv_buffers;

#define HALO_EXCHANGE_FUSED_MANUAL_FUSER_SETUP \
  struct ptr_holder { \
    Real_ptr buffer; \
    Int_ptr  list; \
    Real_ptr var; \
  }; \
  ptr_holder* pack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* pack_lens        = new Index_type[num_neighbors * num_vars]; \
  ptr_holder* unpack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* unpack_lens        = new Index_type[num_neighbors * num_vars];

#define HALO_EXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN \
  delete[] pack_ptr_holders; \
  delete[] pack_lens; \
  delete[] unpack_ptr_holders; \
  delete[] unpack_lens;


#define HALO_EXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_SETUP \
  auto make_pack_lambda = [](Real_ptr buffer, Int_ptr list, Real_ptr var) { \
    return [=](Index_type i) { \
      HALO_PACK_BODY; \
    }; \
  }; \
  using pack_lambda_type = decltype(make_pack_lambda(Real_ptr(), Int_ptr(), Real_ptr())); \
  pack_lambda_type* pack_lambdas = reinterpret_cast<pack_lambda_type*>( \
      malloc(sizeof(pack_lambda_type) * (num_neighbors * num_vars))); \
  Index_type* pack_lens = new Index_type[num_neighbors * num_vars]; \
  auto make_unpack_lambda = [](Real_ptr buffer, Int_ptr list, Real_ptr var) { \
    return [=](Index_type i) { \
      HALO_UNPACK_BODY; \
    }; \
  }; \
  using unpack_lambda_type = decltype(make_unpack_lambda(Real_ptr(), Int_ptr(), Real_ptr())); \
  unpack_lambda_type* unpack_lambdas = reinterpret_cast<unpack_lambda_type*>( \
      malloc(sizeof(unpack_lambda_type) * (num_neighbors * num_vars))); \
  Index_type* unpack_lens = new Index_type[num_neighbors * num_vars];

#define HALO_EXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN \
  free(pack_lambdas); \
  delete[] pack_lens; \
  free(unpack_lambdas); \
  delete[] unpack_lens;


#include "HALO_base.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

namespace rajaperf
{
namespace comm
{

class HALO_EXCHANGE_FUSED : public HALO_base
{
public:

  HALO_EXCHANGE_FUSED(const RunParams& params);

  ~HALO_EXCHANGE_FUSED();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void setOpenMPTuningDefinitions(VariantID vid);
  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setOpenMPTargetTuningDefinitions(VariantID vid);

  void runSeqVariantDirect(VariantID vid);
  void runOpenMPVariantDirect(VariantID vid);
  void runOpenMPTargetVariantDirect(VariantID vid);
  template < size_t block_size >
  void runCudaVariantDirect(VariantID vid);
  template < size_t block_size >
  void runHipVariantDirect(VariantID vid);

  template < typename dispatch_helper >
  void runSeqVariantWorkGroup(VariantID vid);
  template < typename dispatch_helper >
  void runOpenMPVariantWorkGroup(VariantID vid);
  template < typename dispatch_helper >
  void runOpenMPTargetVariantWorkGroup(VariantID vid);
  template < size_t block_size, typename dispatch_helper >
  void runCudaVariantWorkGroup(VariantID vid);
  template < size_t block_size, typename dispatch_helper >
  void runHipVariantWorkGroup(VariantID vid);

private:
  static const size_t default_gpu_block_size = 1024;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  int m_mpi_size = -1;
  int m_my_mpi_rank = -1;
  std::array<int, 3> m_mpi_dims = {-1, -1, -1};

  Index_type m_num_vars;
  Index_type m_var_size;

  std::vector<Real_ptr> m_vars;

  std::vector<Real_ptr> m_pack_buffers;
  std::vector<Real_ptr> m_unpack_buffers;

  std::vector<Real_ptr> m_send_buffers;
  std::vector<Real_ptr> m_recv_buffers;
};

} // end namespace comm
} // end namespace rajaperf

#endif
#endif // closing endif for header file include guard
