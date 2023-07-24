//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define threads per team for target execution (unused)
  //
//const size_t threads_per_team = 256;

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_OMP_TARGET \
  void** pack_ptrs; \
  allocData(DataSpace::OmpTarget, pack_ptrs, 4 * num_neighbors * num_vars); \
  Real_ptr*   pack_buffer_ptrs = reinterpret_cast<Real_ptr*>(pack_ptrs) + 0 * num_neighbors * num_vars; \
  Int_ptr*    pack_list_ptrs   = reinterpret_cast<Int_ptr*>(pack_ptrs) + 1 * num_neighbors * num_vars; \
  Real_ptr*   pack_var_ptrs    = reinterpret_cast<Real_ptr*>(pack_ptrs) + 2 * num_neighbors * num_vars; \
  Index_type* pack_len_ptrs    = reinterpret_cast<Index_type*>(pack_ptrs) + 3 * num_neighbors * num_vars; \
  void** h_pack_ptrs = new void*[4 * num_neighbors * num_vars]; \
  Real_ptr*   h_pack_buffer_ptrs = reinterpret_cast<Real_ptr*>(h_pack_ptrs) + 0 * num_neighbors * num_vars; \
  Int_ptr*    h_pack_list_ptrs   = reinterpret_cast<Int_ptr*>(h_pack_ptrs) + 1 * num_neighbors * num_vars; \
  Real_ptr*   h_pack_var_ptrs    = reinterpret_cast<Real_ptr*>(h_pack_ptrs) + 2 * num_neighbors * num_vars; \
  Index_type* h_pack_len_ptrs    = reinterpret_cast<Index_type*>(h_pack_ptrs) + 3 * num_neighbors * num_vars; \
  void** unpack_ptrs; \
  allocData(DataSpace::OmpTarget, unpack_ptrs, 4 * num_neighbors * num_vars); \
  Real_ptr*   unpack_buffer_ptrs = reinterpret_cast<Real_ptr*>(unpack_ptrs) + 0 * num_neighbors * num_vars; \
  Int_ptr*    unpack_list_ptrs   = reinterpret_cast<Int_ptr*>(unpack_ptrs) + 1 * num_neighbors * num_vars; \
  Real_ptr*   unpack_var_ptrs    = reinterpret_cast<Real_ptr*>(unpack_ptrs) + 2 * num_neighbors * num_vars; \
  Index_type* unpack_len_ptrs    = reinterpret_cast<Index_type*>(unpack_ptrs) + 3 * num_neighbors * num_vars; \
  void** h_unpack_ptrs = new void*[4 * num_neighbors * num_vars]; \
  Real_ptr*   h_unpack_buffer_ptrs = reinterpret_cast<Real_ptr*>(h_unpack_ptrs) + 0 * num_neighbors * num_vars; \
  Int_ptr*    h_unpack_list_ptrs   = reinterpret_cast<Int_ptr*>(h_unpack_ptrs) + 1 * num_neighbors * num_vars; \
  Real_ptr*   h_unpack_var_ptrs    = reinterpret_cast<Real_ptr*>(h_unpack_ptrs) + 2 * num_neighbors * num_vars; \
  Index_type* h_unpack_len_ptrs    = reinterpret_cast<Index_type*>(h_unpack_ptrs) + 3 * num_neighbors * num_vars;

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_COPY_PACK_OMP_TARGET \
  initOpenMPDeviceData(pack_ptrs, h_pack_ptrs, 4 * num_neighbors * num_vars);

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_COPY_UNPACK_OMP_TARGET \
  initOpenMPDeviceData(unpack_ptrs, h_unpack_ptrs, 4 * num_neighbors * num_vars);

#define HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_OMP_TARGET \
  deallocData(DataSpace::OmpTarget, pack_ptrs); \
  delete[] h_pack_ptrs; \
  deallocData(DataSpace::OmpTarget, unpack_ptrs); \
  delete[] h_unpack_ptrs;


void HALOEXCHANGE_FUSED::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_FUSED_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP_OMP_TARGET;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type pack_index = 0;
      Index_type pack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type len = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          h_pack_buffer_ptrs[pack_index] = buffer;
          h_pack_list_ptrs[pack_index] = list;
          h_pack_var_ptrs[pack_index] = var;
          h_pack_len_ptrs[pack_index] = len;
          pack_len_sum += len;
          pack_index += 1;
          buffer += len;
        }
      }
      HALOEXCHANGE_FUSED_MANUAL_FUSER_COPY_PACK_OMP_TARGET;
      Index_type pack_len_ave = (pack_len_sum + pack_index-1) / pack_index;
      #pragma omp target is_device_ptr(pack_buffer_ptrs, pack_list_ptrs, pack_var_ptrs, pack_len_ptrs) device( did )
      #pragma omp teams distribute parallel for collapse(2) schedule(static, 1)
      for (Index_type j = 0; j < pack_index; j++) {
        for (Index_type ii = 0; ii < pack_len_ave; ii++) {

          Real_ptr   buffer = pack_buffer_ptrs[j];
          Int_ptr    list   = pack_list_ptrs[j];
          Real_ptr   var    = pack_var_ptrs[j];
          Index_type len    = pack_len_ptrs[j];

          for (Index_type i = ii; i < len; i += pack_len_ave) {
            HALOEXCHANGE_FUSED_PACK_BODY;
          }
        }
      }

      Index_type unpack_index = 0;
      Index_type unpack_len_sum = 0;

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          h_unpack_buffer_ptrs[unpack_index] = buffer;
          h_unpack_list_ptrs[unpack_index] = list;
          h_unpack_var_ptrs[unpack_index] = var;
          h_unpack_len_ptrs[unpack_index] = len;
          unpack_len_sum += len;
          unpack_index += 1;
          buffer += len;
        }
      }
      HALOEXCHANGE_FUSED_MANUAL_FUSER_COPY_UNPACK_OMP_TARGET;
      Index_type unpack_len_ave = (unpack_len_sum + unpack_index-1) / unpack_index;
      #pragma omp target is_device_ptr(unpack_buffer_ptrs, unpack_list_ptrs, unpack_var_ptrs, unpack_len_ptrs) device( did )
      #pragma omp teams distribute parallel for collapse(2) schedule(static, 1)
      for (Index_type j = 0; j < unpack_index; j++) {
        for (Index_type ii = 0; ii < unpack_len_ave; ii++) {

          Real_ptr   buffer = unpack_buffer_ptrs[j];
          Int_ptr    list   = unpack_list_ptrs[j];
          Real_ptr   var    = unpack_var_ptrs[j];
          Index_type len    = unpack_len_ptrs[j];

          for (Index_type i = ii; i < len; i += unpack_len_ave) {
            HALOEXCHANGE_FUSED_UNPACK_BODY;
          }
        }
      }

    }
    stopTimer();

    HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN_OMP_TARGET;

  } else if ( vid == RAJA_OpenMPTarget ) {

    using AllocatorHolder = RAJAPoolAllocatorHolder<
        RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator>>;
    using Allocator = AllocatorHolder::Allocator<char>;

    AllocatorHolder allocatorHolder;

    using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::omp_target_work /*<threads_per_team>*/,
                                 RAJA::ordered,
                                 RAJA::constant_stride_array_of_objects >;

    using workpool = RAJA::WorkPool< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    using workgroup = RAJA::WorkGroup< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

    using worksite = RAJA::WorkSite< workgroup_policy,
                                     Index_type,
                                     RAJA::xargs<>,
                                     Allocator >;

    workpool pool_pack  (allocatorHolder.template getAllocator<char>());
    workpool pool_unpack(allocatorHolder.template getAllocator<char>());
    pool_pack.reserve(num_neighbors * num_vars, 1024ull*1024ull);
    pool_unpack.reserve(num_neighbors * num_vars, 1024ull*1024ull);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_fused_pack_base_lam = [=](Index_type i) {
                HALOEXCHANGE_FUSED_PACK_BODY;
              };
          pool_pack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_pack_base_lam );
          buffer += len;
        }
      }
      workgroup group_pack = pool_pack.instantiate();
      worksite site_pack = group_pack.run();

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type  len  = unpack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_fused_unpack_base_lam = [=](Index_type i) {
                HALOEXCHANGE_FUSED_UNPACK_BODY;
              };
          pool_unpack.enqueue(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_fused_unpack_base_lam );
          buffer += len;
        }
      }
      workgroup group_unpack = pool_unpack.instantiate();
      worksite site_unpack = group_unpack.run();

    }
    stopTimer();

  } else {
     getCout() << "\n HALOEXCHANGE_FUSED : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
