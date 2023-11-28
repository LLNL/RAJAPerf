//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALO_PACKING_FUSED kernel reference implementation:
///
/// // pack buffers for neighbors
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
/// // unpack buffers for neighbors
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

#ifndef RAJAPerf_Comm_HALO_PACKING_FUSED_HPP
#define RAJAPerf_Comm_HALO_PACKING_FUSED_HPP

#define HALO_PACKING_FUSED_DATA_SETUP \
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

#define HALO_PACKING_FUSED_MANUAL_FUSER_SETUP \
  struct ptr_holder { \
    Real_ptr buffer; \
    Int_ptr  list; \
    Real_ptr var; \
  }; \
  ptr_holder* pack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* pack_lens        = new Index_type[num_neighbors * num_vars]; \
  ptr_holder* unpack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* unpack_lens        = new Index_type[num_neighbors * num_vars];

#define HALO_PACKING_FUSED_MANUAL_FUSER_TEARDOWN \
  delete[] pack_ptr_holders; \
  delete[] pack_lens; \
  delete[] unpack_ptr_holders; \
  delete[] unpack_lens;


#define HALO_PACKING_FUSED_MANUAL_LAMBDA_FUSER_SETUP \
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

#define HALO_PACKING_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN \
  free(pack_lambdas); \
  delete[] pack_lens; \
  free(unpack_lambdas); \
  delete[] unpack_lens;


#include "HALO_base.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
namespace comm
{

class HALO_PACKING_FUSED : public HALO_base
{
public:

  HALO_PACKING_FUSED(const RunParams& params);

  ~HALO_PACKING_FUSED();

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
  static const size_t default_gpu_block_size = 1024;
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
