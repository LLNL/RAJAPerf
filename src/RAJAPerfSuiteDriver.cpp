//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifdef RUN_KOKKOS
#include <Kokkos_Core.hpp>
#endif

#include "common/Executor.hpp"

#include <iostream>

#ifdef RAJA_PERFSUITE_ENABLE_MPI
#include <mpi.h>
#endif

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
#ifdef RAJA_PERFSUITE_ENABLE_MPI
  MPI_Init(&argc, &argv);

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  rajaperf::getCout() << "\n\nRunning with " << num_ranks << " MPI ranks..." << std::endl;
#endif
#ifdef RUN_KOKKOS
  Kokkos::initialize(argc, argv);
#endif

  // STEP 1: Create suite executor object
  rajaperf::Executor executor(argc, argv);

  // STEP 2: Assemble kernels and variants to run
  executor.setupSuite();

  // STEP 3: Report suite run summary
  //         (enable users to catch errors before entire suite is run)
  executor.reportRunSummary(rajaperf::getCout());

  // STEP 4: Execute suite
  executor.runSuite();

  // STEP 5: Generate suite execution reports
  executor.outputRunData();

  rajaperf::getCout() << "\n\nDONE!!!...." << std::endl;

#ifdef RUN_KOKKOS
  Kokkos::finalize();
#endif
#ifdef RAJA_PERFSUITE_ENABLE_MPI
  MPI_Finalize();
#endif

  return 0;
}
