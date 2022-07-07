//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALOEXCHANGE_FUSED kernel reference implementation:
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
///       HALOEXCHANGE_FUSED_PACK_BODY;
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
///       HALOEXCHANGE_FUSED_UNPACK_BODY;
///     }
///     buffer += len;
///   }
/// }
///

#ifndef RAJAPerf_Apps_HALOEXCHANGE_FUSED_HPP
#define RAJAPerf_Apps_HALOEXCHANGE_FUSED_HPP

#define HALOEXCHANGE_FUSED_DATA_SETUP \
  std::vector<Real_ptr> vars = m_vars; \
  std::vector<Real_ptr> buffers = m_buffers; \
\
  Index_type num_neighbors = s_num_neighbors; \
  Index_type num_vars = m_num_vars; \
  std::vector<Int_ptr> pack_index_lists = m_pack_index_lists; \
  std::vector<Index_type> pack_index_list_lengths = m_pack_index_list_lengths; \
  std::vector<Int_ptr> unpack_index_lists = m_unpack_index_lists; \
  std::vector<Index_type> unpack_index_list_lengths = m_unpack_index_list_lengths;

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP \
  struct ptr_holder { \
    Real_ptr buffer; \
    Int_ptr  list; \
    Real_ptr var; \
  }; \
  ptr_holder* pack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* pack_lens        = new Index_type[num_neighbors * num_vars]; \
  ptr_holder* unpack_ptr_holders = new ptr_holder[num_neighbors * num_vars]; \
  Index_type* unpack_lens        = new Index_type[num_neighbors * num_vars];

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN \
  delete[] pack_ptr_holders; \
  delete[] pack_lens; \
  delete[] unpack_ptr_holders; \
  delete[] unpack_lens;

#define HALOEXCHANGE_FUSED_PACK_BODY \
  buffer[i] = var[list[i]];

#define HALOEXCHANGE_FUSED_UNPACK_BODY \
  var[list[i]] = buffer[i];


#define HALOEXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_SETUP \
  auto make_pack_lambda = [](Real_ptr buffer, Int_ptr list, Real_ptr var) { \
    return [=](Index_type i) { \
      HALOEXCHANGE_FUSED_PACK_BODY; \
    }; \
  }; \
  using pack_lambda_type = decltype(make_pack_lambda(Real_ptr(), Int_ptr(), Real_ptr())); \
  pack_lambda_type* pack_lambdas = reinterpret_cast<pack_lambda_type*>( \
      malloc(sizeof(pack_lambda_type) * (num_neighbors * num_vars))); \
  Index_type* pack_lens = new Index_type[num_neighbors * num_vars]; \
  auto make_unpack_lambda = [](Real_ptr buffer, Int_ptr list, Real_ptr var) { \
    return [=](Index_type i) { \
      HALOEXCHANGE_FUSED_UNPACK_BODY; \
    }; \
  }; \
  using unpack_lambda_type = decltype(make_unpack_lambda(Real_ptr(), Int_ptr(), Real_ptr())); \
  unpack_lambda_type* unpack_lambdas = reinterpret_cast<unpack_lambda_type*>( \
      malloc(sizeof(unpack_lambda_type) * (num_neighbors * num_vars))); \
  Index_type* unpack_lens = new Index_type[num_neighbors * num_vars];

#define HALOEXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN \
  free(pack_lambdas); \
  delete[] pack_lens; \
  free(unpack_lambdas); \
  delete[] unpack_lens;


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
class RunParams;

namespace apps
{

class HALOEXCHANGE_FUSED : public KernelBase
{
public:

  HALOEXCHANGE_FUSED(const RunParams& params);

  ~HALOEXCHANGE_FUSED();

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
  static const size_t default_gpu_block_size = 1024;
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
