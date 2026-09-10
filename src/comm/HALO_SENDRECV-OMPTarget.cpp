//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_SENDRECV.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI) && defined(RAJA_ENABLE_TARGET_OPENMP)

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


void HALO_SENDRECV::runOpenMPTargetVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  HALO_SENDRECV_DATA_SETUP;

  if ( vid == Base_OpenMPTarget ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("HALO_SENDRECV_1");
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
      RP_CALI_SUBKERNEL_END("HALO_SENDRECV_1");

    }
    stopTimer();

  } else {
     getCout() << "\n HALO_SENDRECV : Unknown OMP Target variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HALO_SENDRECV, OpenMPTarget, Base_OpenMPTarget)

} // end namespace comm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_TARGET_OPENMP
