//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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


void HALO_PACKING::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  HALO_PACKING_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_pack_k");
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              HALO_PACK_BODY;
            }
            RP_CALI_SUBKERNEL_END("HALO_PACKING_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_unpack_k");
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              HALO_UNPACK_BODY;
            }
            RP_CALI_SUBKERNEL_END("HALO_PACKING_unpack_k");
            buffer += len;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_pack_k");
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              halo_packing_pack_base_lam(i);
            }
            RP_CALI_SUBKERNEL_END("HALO_PACKING_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_unpack_k");
            #pragma omp parallel for
            for (Index_type i = 0; i < len; i++) {
              halo_packing_unpack_base_lam(i);
            }
            RP_CALI_SUBKERNEL_END("HALO_PACKING_unpack_k");
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
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_pack_k");
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_packing_pack_base_lam );
            RP_CALI_SUBKERNEL_END("HALO_PACKING_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            res.memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            res.memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_packing_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_PACKING_unpack_k");
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_packing_unpack_base_lam );
            RP_CALI_SUBKERNEL_END("HALO_PACKING_unpack_k");
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

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HALO_PACKING, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace comm
} // end namespace rajaperf
