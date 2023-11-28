//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOPACKING.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#include "common/OpenMPTargetDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{

  //
  // Define threads per team for target execution
  //
  const size_t threads_per_team = 256;


void HALOPACKING::runOpenMPTargetVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  HALOPACKING_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          #pragma omp target is_device_ptr(buffer, list, var) device( did )
          #pragma omp teams distribute parallel for schedule(static, 1)
          for (Index_type i = 0; i < len; i++) {
            HALO_PACK_BODY;
          }
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          copyData(dataSpace, unpack_buffers[l],
                   DataSpace::Host, recv_buffers[l],
                   len*num_vars);
        }

        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          #pragma omp target is_device_ptr(buffer, list, var) device( did )
          #pragma omp teams distribute parallel for schedule(static, 1)
          for (Index_type i = 0; i < len; i++) {
            HALO_UNPACK_BODY;
          }
          buffer += len;
        }
      }

    }
    stopTimer();

  } else if ( vid == RAJA_OpenMPTarget ) {

    using EXEC_POL = RAJA::omp_target_parallel_for_exec<threads_per_team>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = pack_buffers[l];
        Int_ptr list = pack_index_lists[l];
        Index_type  len  = pack_index_list_lengths[l];
        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_pack_base_lam = [=](Index_type i) {
                HALO_PACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_pack_base_lam );
          buffer += len;
        }

        if (separate_buffers) {
          copyData(DataSpace::Host, send_buffers[l],
                   dataSpace, pack_buffers[l],
                   len*num_vars);
        }
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Real_ptr buffer = unpack_buffers[l];
        Int_ptr list = unpack_index_lists[l];
        Index_type len = unpack_index_list_lengths[l];
        if (separate_buffers) {
          copyData(dataSpace, unpack_buffers[l],
                   DataSpace::Host, recv_buffers[l],
                   len*num_vars);
        }

        for (Index_type v = 0; v < num_vars; ++v) {
          Real_ptr var = vars[v];
          auto haloexchange_unpack_base_lam = [=](Index_type i) {
                HALO_UNPACK_BODY;
              };
          RAJA::forall<EXEC_POL>(
              RAJA::TypedRangeSegment<Index_type>(0, len),
              haloexchange_unpack_base_lam );
          buffer += len;
        }
      }

    }
    stopTimer();

  } else {
     getCout() << "\n HALOPACKING : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
