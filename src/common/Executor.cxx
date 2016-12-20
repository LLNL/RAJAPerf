/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for executor class that runs suite.
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


#include "Executor.hxx"

#include "common/RAJAPerfSuite.hxx"

namespace rajaperf {

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv)
{
}

Executor::~Executor()
{
}

void Executor::setupSuite()
{
  //
  // Assemble kernels to execute
  //
  if ( run_params.kernel_filter.size() == 0 ) {

    //
    // No kernels specified in input options, run them all...
    //
    for (int ikern = 0; ikern < NUM_KERNELS; ++ikern) {
      kernels.push_back( getKernelObject(static_cast<KernelID>(ikern)) );
    }

  } else {

     //
     // Determine which kernels to run based on provided input options.
     //
     // These are strings in run_params.run_kernels
     //

  } 

  //
  // Assemble variants to execute
  //
  if ( run_params.variant_filter.size() == 0 ) {

    //
    // No variants specified in input options, run them all...
    //
    for (int ivar = 0; ivar < NUM_VARIANTS; ++ivar) {
      variants.push_back( static_cast<VariantID>(ivar) );
    }

  } else {

     //
     // Determine which variants to run based on provided input options.
     //
     // These are strings in run_params.run_variants
     //

  }
}

void Executor::reportRunSummary()
{
}

void Executor::runSuite()
{
}

void Executor::outputRunData()
{
}

}  // closing brace for rajaperf namespace
