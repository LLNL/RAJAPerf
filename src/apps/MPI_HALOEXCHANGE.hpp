//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// MPI_HALOEXCHANGE kernel reference implementation:
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

#ifndef RAJAPerf_Apps_MPI_HALOEXCHANGE_HPP
#define RAJAPerf_Apps_MPI_HALOEXCHANGE_HPP

#define MPI_HALOEXCHANGE_DATA_SETUP \
  HALOEXCHANGE_base_DATA_SETUP \
  \
  const int my_mpi_rank = m_my_mpi_rank; \
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


#include "HALOEXCHANGE_base.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#include <vector>

namespace rajaperf
{
namespace apps
{

class MPI_HALOEXCHANGE : public HALOEXCHANGE_base
{
public:

  MPI_HALOEXCHANGE(const RunParams& params);

  ~MPI_HALOEXCHANGE();

  void setUp(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  int m_my_mpi_rank = -1;
  std::vector<int> m_mpi_ranks;

  std::vector<Real_ptr> m_pack_buffers;
  std::vector<Real_ptr> m_unpack_buffers;

  std::vector<Real_ptr> m_send_buffers;
  std::vector<Real_ptr> m_recv_buffers;
};

} // end namespace apps
} // end namespace rajaperf

#endif
#endif // closing endif for header file include guard
