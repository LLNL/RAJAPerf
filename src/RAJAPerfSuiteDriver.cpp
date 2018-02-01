//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
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
