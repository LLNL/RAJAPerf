//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_PACKING.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{


void HALO_PACKING::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  HALO_PACKING_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            #pragma omp parallel for
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
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              HALO_UNPACK_BODY;
            }
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              halo_packing_pack_base_lam(i);
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
            auto halo_packing_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              halo_packing_unpack_base_lam(i);
            }
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      using EXEC_POL = RAJA::omp_parallel_for_exec;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_packing_pack_base_lam );
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
            auto halo_packing_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_packing_unpack_base_lam );
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n HALO_PACKING : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace comm
} // end namespace rajaperf
