/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file that drives performance suite.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "common/RAJAPerfSuite.hxx"

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // STEP 0: Parse command line options and store in params object
  rajaperf::RunParams params(argc, argv);

#if 0
  // STEP 1: Report parameter summary
  rajaperf::reportRunSummary(params);  
    
  // STEP 2: Run the loop suite
  rajaperf::Executor executor(params);
  executor.run();

  // STEP 3: Write execution reports
  rajaperf::outputRunData(params);  
#endif

  return 0;
}
