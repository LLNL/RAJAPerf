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

#include <list>
#include <set>
#include <vector>
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

  if ( !run_params.goodToGo() ) {
    return;
  }

  typedef std::list<std::string> Slist;
  typedef std::vector<std::string> Svector;
  typedef std::set<std::string> Sset;
  typedef std::set<KernelID> KIDset;
  typedef std::set<VariantID> VIDset;

  //
  // Assemble kernels to execute from input filter.
  //
  // unmatched will contain unmatched input.
  // run_kern will be non-duplicated ordered set of IDs of kernel to run.
  //
  const Svector& kernel_filter = run_params.getKernelFilter();
  Slist unmatched;

  KIDset run_kern;

  if ( kernel_filter.empty() ) {

    //
    // No kernels specified in input, run them all...
    //
    for (size_t ik = 0; ik < NumKernels; ++ik) {
      run_kern.insert( static_cast<KernelID>(ik) );
    }

  } else {

    //
    // Need to parse input to determine which kernels to run
    // 

    // Make list copy of input kernel filter to manipulate
    Slist filter(kernel_filter.begin(), kernel_filter.end());

    //
    // Search input for matching suite names.
    // suites2run will contain names of suites to run.
    //
    Svector suites2run;
    for (Slist::iterator fit = filter.begin(); fit != filter.end(); ++fit) {
      for (size_t is = 0; is < NumSuites; ++is) {
        const std::string& suite_name = getSuiteName(static_cast<SuiteID>(is));
        if ( suite_name == *fit ) {
          suites2run.push_back(suite_name);
        }
      }
    }

    // 
    // If suite name(s) found in input, assemble kernels in suite(s) 
    // to run and remove those suite name(s) from input filter.
    //
    for (size_t is = 0; is < suites2run.size(); ++is) {
      const std::string& sname(suites2run[is]);

      for (size_t ik = 0; ik < NumKernels; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(sname) != std::string::npos ) {
          run_kern.insert(kid);
        }
      }

      filter.remove(sname);
    }

    //
    // If filter still has entries, look for matching names of individual 
    // kernels in remaining input.
    // 
    // Assemble unmatched input for warning message.
    //

    for (Slist::iterator fit = filter.begin(); fit != filter.end(); ++fit) {
      bool found_it = false;

      for (size_t ik = 0; ik < NumKernels && !found_it; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(*fit) != std::string::npos ) {
          run_kern.insert(kid);
          found_it = true;
        }
      }

      if ( !found_it )  unmatched.push_back(*fit); 
    }

  } 

  //
  // Create kernel objects to execute. If unmatched is not empty, 
  // then there was unmatched input --> report error.
  //
  if ( unmatched.empty() ) {

    for (KIDset::iterator kid = run_kern.begin(); 
         kid != run_kern.end(); ++kid) {
      kernels.push_back( getKernelObject(*kid, run_params) );
    }

  } else {
    run_params.setGoodToGo(false); 
#if 0 // RDH TODO
    Report unknown input in kernel filter in Run Summary...
#else
    std::cout << "\nBAD KERNEL INPUT!!\n";
#endif
  }


  //
  // Assemble variants to execute from input filter.
  //
  // unmatched will contain unmatched input.
  // run_var will be non-duplicated ordered set of IDs of variants to run.
  //
  const Svector& variant_filter = run_params.getVariantFilter();
  unmatched.clear();

  VIDset run_var;

  if ( variant_filter.empty() ) {

    //
    // No variants specified in input options, run them all...
    //
    for (size_t iv = 0; iv < NumVariants; ++iv) {
      run_var.insert( static_cast<VariantID>(iv) );
    }

  } else {

    //
    // Need to parse input to determine which variants to run
    // 

    // Make list copy of input variant filter to manipulate
    Slist filter(variant_filter.begin(), variant_filter.end());

    //
    // Search input for matching variant names.
    //
    // Assemble unmatched input for warning message.
    //
    Slist unmatched;

    for (Slist::iterator fit = filter.begin(); fit != filter.end(); ++fit) {
      bool found_it = false;

      for (size_t iv = 0; iv < NumVariants && !found_it; ++iv) {
        VariantID vid = static_cast<VariantID>(iv);
        if ( getVariantName(vid) == *fit ) {
          run_var.insert(vid);
          found_it = true;
        }
      }

      if ( !found_it )  unmatched.push_back(*fit);
    }

  }

  //
  // Set variants to execute. If unmatched is not empty, 
  // then there was unmatched input --> report error.
  //
  if ( unmatched.empty() ) {

    for (VIDset::iterator vid = run_var.begin();
         vid != run_var.end(); ++vid) {
      variants.push_back( *vid );
    }

  } else {
    run_params.setGoodToGo(false);
#if 0 // RDH TODO
    Report unknown input in variant filter in Run Summary...
#else
    std::cout << "\nBAD VARIANT INPUT!!\n";
#endif
  }

}

void Executor::reportRunSummary()
{
  if ( !run_params.goodToGo() ) {
#if 0 // RDH TODO
    Report bad input if it exists, else report that suite will not run
    due to info request
#endif
    std::cout << "\n RAJA perf suite will not be run now due to bad input"
              << " or info request..." << std::endl;
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
  std::cout << "\nExecutor::runSuite : \n";
  if ( !run_params.goodToGo() ) {
    return;
  }

  std::cout << "\nRUNNING!\n";
  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {  
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      for (size_t iv = 0; iv < variants.size(); ++iv) {
         kernels[ik]->execute( variants[iv] );
      } 
    }
  }
}

void Executor::outputRunData()
{
  if ( !run_params.goodToGo() ) {
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
