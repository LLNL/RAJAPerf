//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "common/Executor.hpp"
#include "common/QuickKernelBase.hpp"
#include <iostream>

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // Create suite executor object with the arguments that were passed in
  // rajaperf::Executor executor(argc, argv);

#if defined(RUN_KOKKOS)
            Kokkos::initialize(argc, argv);
#endif // RUN_KOKKOS

  rajaperf::Executor executor(argc, argv);
  rajaperf::make_perfsuite_executor(&executor, argc, argv);
  
  // Assemble kernels and variants to run
  executor.setupSuite();

  // Report suite run summary 
  //         (enable users to catch errors before entire suite is run)
  executor.reportRunSummary(std::cout); 

  // Execute suite of selected tests
  executor.runSuite();

  // Generate suite execution reports
  executor.outputRunData();

#if defined(RUN_KOKKOS)
        Kokkos::finalize(); // TODO DZP: should this be here?  Good question.  AJP
#endif

  std::cout << "\n\nDONE!!!...." << std::endl; 

  return 0;
}
