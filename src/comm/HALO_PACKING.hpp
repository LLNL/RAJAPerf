//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALO_PACKING kernel reference implementation:
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
/// }
///
/// // unpack a buffer for each neighbor
/// for (Index_type l = 0; l < num_neighbors; ++l) {
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

#ifndef RAJAPerf_Comm_HALO_PACKING_HPP
#define RAJAPerf_Comm_HALO_PACKING_HPP

#define HALO_PACKING_DATA_SETUP \
  HALO_BASE_DATA_SETUP \
  \
  Index_type num_vars = m_num_vars; \
  std::vector<Real_ptr> vars = m_vars; \
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


#include "HALO_base.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
namespace comm
{

class HALO_PACKING : public HALO_base
{
public:

  HALO_PACKING(const RunParams& params);

  ~HALO_PACKING();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
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

#endif // closing endif for header file include guard
