//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "common/Executor.hpp"

#include <iostream>

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // STEP 1: Create suite executor object
  rajaperf::Executor executor(argc, argv);

  // STEP 2: Assemble kernels and variants to run
  executor.setupSuite();

  // STEP 3: Report suite run summary 
  //         (enable users to catch errors before entire suite is run)
  executor.reportRunSummary(std::cout); 

  // STEP 4: Execute suite
  executor.runSuite();

  // STEP 5: Generate suite execution reports
  executor.outputRunData();

  std::cout << "\n\nDONE!!!...." << std::endl; 

  return 0;
}
