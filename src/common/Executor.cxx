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
#include "common/KernelBase.hxx"

#include <iostream>

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
  if ( !run_params.good2go ) {
    return;
  }

  //
  // Assemble kernels to execute
  //
  if ( run_params.kernel_filter.size() == 0 ) {

    //
    // No kernels specified in input options, run them all...
    //
    for (int ikern = 0; ikern < NumKernels; ++ikern) {
      kernels.push_back( getKernelObject(static_cast<KernelID>(ikern),
                                         run_params.sample_fraction,
                                         run_params.size_fraction) );
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
    for (int ivar = 0; ivar < NumVariants; ++ivar) {
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
  if ( !run_params.good2go ) {
    std::cout << "\n\n RAJA perf suite will not be run now due to bad input"              << " or info request..." << std::endl;
    std::cout.flush();
    return;
  } else {
   // 
   // Generate formatted summary of suite execution:
   //   - system, date, and time (e.g., utilities in ctime)
   //   - Compiler, version, and options 
   //       (RDH: I have something to generate this info in LCALS)
   //   - RunParams: npasses, sample_fraction, size_fraction 
   //       (in RunParams object)
   //   - Listing of names of kernels and variants that will be run
   //       (easily obtained from kernels and variants data members)
   //
   // Send to stdout and also to output summary file....
   //
  }
}

void Executor::runSuite()
{
  if ( !run_params.good2go ) {
    return;
  }

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    for (size_t iv = 0; iv < variants.size(); ++iv) {
       kernels[ik]->execute( variants[iv] );
    } 
  }
}

void Executor::outputRunData()
{
  if ( !run_params.good2go ) {
    return;
  }

  //
  // (RDH: I have code to generate this info in LCALS -- just need to
  //       pull it out and massage based on what we want)
  //
  // Generate output in appropriate format and write to file(s) in
  // appropriate format for what we want (e.g., csv (for tools), 
  // easy-to-read (for humans)), etc.: 
  //   - execution timings (max/min/avg) for each kernel and variant
  //     (note: if npasses == 1, these are the same so only report time)
  //   - speedup for each kernel variant relative to baseline 
  //     (or something else?)
  //   - run samples and run size information for each kernel
  //   - we should think about defining a FOM for the entire suite. 
  //     I did this for LCALS since it was needed for CORAL. I debated this
  //     with numerous folks and wasn't very satisfied what I came up with,
  //     but it seemed like a reasonable way to generate a single number
  //     with which to compare results of different configurations.
  //
}

}  // closing brace for rajaperf namespace
