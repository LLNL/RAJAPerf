//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_SENDRECV.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI) && defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace comm
{


void HALO_SENDRECV::runHipVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  const Index_type run_reps = getRunReps();

  HALO_SENDRECV_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = unpack_index_list_lengths[l];
        MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
      }

      for (Index_type l = 0; l < num_neighbors; ++l) {
        Index_type len = pack_index_list_lengths[l];
        MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
            mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
      }

      MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

    }
    stopTimer();

  } else {
     getCout() << "\n HALO_SENDRECV : Unknown Hip variant id = " << vid << std::endl;
  }
}

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
