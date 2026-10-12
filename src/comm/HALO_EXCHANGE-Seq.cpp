//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_EXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#include <iostream>

namespace rajaperf
{
namespace comm
{


void HALO_EXCHANGE::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  HALO_EXCHANGE_DATA_SETUP;

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_pack_k");
            for (Index_type i = 0; i < len; i++) {
              HALO_PACK_BODY;
            }
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }

          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        for (Index_type ll = 0; ll < num_neighbors; ++ll) {
          int l = -1;
          MPI_Waitany(num_neighbors, unpack_mpi_requests.data(), &l, MPI_STATUS_IGNORE);

          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_unpack_k");
            for (Index_type i = 0; i < len; i++) {
              HALO_UNPACK_BODY;
            }
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_unpack_k");
            buffer += len;
          }
        }

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_exchange_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_pack_k");
            for (Index_type i = 0; i < len; i++) {
              halo_exchange_pack_base_lam(i);
            }
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }

          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        for (Index_type ll = 0; ll < num_neighbors; ++ll) {
          int l = -1;
          MPI_Waitany(num_neighbors, unpack_mpi_requests.data(), &l, MPI_STATUS_IGNORE);

          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_exchange_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_unpack_k");
            for (Index_type i = 0; i < len; i++) {
              halo_exchange_unpack_base_lam(i);
            }
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_unpack_k");
            buffer += len;
          }
        }

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      auto res{getHostResource()};

      using EXEC_POL = RAJA::seq_exec;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_exchange_pack_base_lam = [=](Index_type i) {
                  HALO_PACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_pack_k");
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_exchange_pack_base_lam );
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_pack_k");
            buffer += len;
          }

          if (separate_buffers) {
            res.memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }

          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        for (Index_type ll = 0; ll < num_neighbors; ++ll) {
          int l = -1;
          MPI_Waitany(num_neighbors, unpack_mpi_requests.data(), &l, MPI_STATUS_IGNORE);

          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            res.memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto halo_exchange_unpack_base_lam = [=](Index_type i) {
                  HALO_UNPACK_BODY;
                };
            RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_unpack_k");
            RAJA::forall<EXEC_POL>( res,
                RAJA::TypedRangeSegment<Index_type>(0, len),
                halo_exchange_unpack_base_lam );
            RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_unpack_k");
            buffer += len;
          }
        }

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n HALO_EXCHANGE : Unknown variant id = " << vid << std::endl;
    }

  }

}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HALO_EXCHANGE, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace comm
} // end namespace rajaperf

#endif
