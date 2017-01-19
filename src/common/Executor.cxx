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
  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    delete kernels[ik];
  }
}


void Executor::setupSuite()
{

  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state == RunParams::InfoRequest || in_state == RunParams::BadInput ) {
    return;
  }

  typedef std::list<std::string> Slist;
  typedef std::vector<std::string> Svector;
  typedef std::set<std::string> Sset;
  typedef std::set<KernelID> KIDset;
  typedef std::set<VariantID> VIDset;

  //
  // Determine which kernels to execute from input.
  // run_kern will be non-duplicated ordered set of IDs of kernel to run.
  //
  const Svector& kernel_input = run_params.getKernelInput();

  KIDset run_kern;

  if ( kernel_input.empty() ) {

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

    // Make list copy of kernel input to manipulate
    // (need to process potential group names and/or kernel names)
    Slist input(kernel_input.begin(), kernel_input.end());

    //
    // Search input for matching group names.
    // groups2run will contain names of groups to run.
    //
    Svector groups2run;
    for (Slist::iterator it = input.begin(); it != input.end(); ++it) {
      for (size_t ig = 0; ig < NumGroups; ++ig) {
        const std::string& group_name = getGroupName(static_cast<GroupID>(ig));
        if ( group_name == *it ) {
          groups2run.push_back(group_name);
        }
      }
    }

    // 
    // If group name(s) found in input, assemble kernels in group(s) 
    // to run and remove those group name(s) from input list.
    //
    for (size_t ig = 0; ig < groups2run.size(); ++ig) {
      const std::string& gname(groups2run[ig]);

      for (size_t ik = 0; ik < NumKernels; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(gname) != std::string::npos ) {
          run_kern.insert(kid);
        }
      }

      input.remove(gname);
    }

    //
    // Look for matching names of individual kernels in remaining input.
    // 
    // Assemble unknown input for warning message.
    //
    Svector unknown;

    for (Slist::iterator it = input.begin(); it != input.end(); ++it) {
      bool found_it = false;

      for (size_t ik = 0; ik < NumKernels && !found_it; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(*it) != std::string::npos ) {
          run_kern.insert(kid);
          found_it = true;
        }
      }

      if ( !found_it )  unknown.push_back(*it); 
    }

    run_params.setUnknownKernelInput(unknown);

  }


  //
  // Determine variants to execute from input.
  // run_var will be non-duplicated ordered set of IDs of variants to run.
  //
  const Svector& variant_input = run_params.getVariantInput();

  VIDset run_var;

  if ( variant_input.empty() ) {

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

    //
    // Search input for matching variant names.
    //
    // Assemble unknown input for warning message.
    //
    Svector unknown;

    for (size_t it = 0; it < variant_input.size(); ++it) {
      bool found_it = false;

      for (size_t iv = 0; iv < NumVariants && !found_it; ++iv) {
        VariantID vid = static_cast<VariantID>(iv);
        if ( getVariantName(vid) == variant_input[it] ) {
          run_var.insert(vid);
          found_it = true;
        }
      }

      if ( !found_it )  unknown.push_back(variant_input[it]);
    }

    run_params.setUnknownVariantInput(unknown);

  }

  //
  // Create kernel objects and variants to execute. If unknown input is not 
  // empty for either case, then there were unmatched input items.
  // 
  // A message will be emitted later so user can sort it out...
  //

  if ( !(run_params.getUnknownKernelInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput); 

  } else { // kernel input looks good

    for (KIDset::iterator kid = run_kern.begin(); 
         kid != run_kern.end(); ++kid) {
      kernels.push_back( getKernelObject(*kid, run_params) );
    }

    if ( !(run_params.getUnknownVariantInput().empty()) ) {

       run_params.setInputState(RunParams::BadInput);

    } else { // variant input lools good

      for (VIDset::iterator vid = run_var.begin();
           vid != run_var.end(); ++vid) {
        variants.push_back( *vid );
      }

      //
      // If we've gotten to this point, we have good input to run.
      //
      if ( run_params.getInputState() != RunParams::DryRun ) {
        run_params.setInputState(RunParams::GoodToRun);
      }

    } // kernel and variant input both look good

  } // if kernel input looks good

}


void Executor::reportRunSummary(std::ostream& str) const
{
  RunParams::InputOpt in_state = run_params.getInputState();

  if ( in_state == RunParams::BadInput ) {

    str << "\nRunParams state:\n";
    str <<   "----------------";
    run_params.print(str);

    str << "\n\nRAJA perf suite will not be run now due to bad input."
        << "\n  See run parameters or option messages above.\n" 
        << std::endl;

  } else if ( in_state == RunParams::GoodToRun || 
              in_state == RunParams::DryRun ) { 

    //
    // RDH: Note the following information should also be written 
    //      to a run summary file
    //

    str << "\nRunParams state:";
    str << "\n----------------";
    run_params.print(str);

    // 
    // Generate formatted summary of suite execution:
    //   - system, date, and time (e.g., utilities in ctime)
    //   - Compiler, version, and options 
    //       (RDH: I think I have something to generate this info in LCALS)
    // 

    str << "\n\nRAJA perf suite will run with the following kernels and variants." 
        << std::endl;

    str << "\nKernels"
        << "\n-------\n";
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
       str << kernels[ik]->getName() << std::endl;
    }

    str << "\nVariants"
        << "\n--------\n";
    for (size_t iv = 0; iv < variants.size(); ++iv) {
       str << getVariantName(variants[iv]) << std::endl;
    }

  }

  str.flush();
}

void Executor::runSuite()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::GoodToRun ) {
    return;
  }

  std::cout << "\n\nRunning specified kernels and variants!\n";
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
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::GoodToRun ) {
    return;
  }

  std::cout << "\nOutput data generation not impllemented yet!!!" << std::endl;
  std::cout.flush();

  //
  // (RDH: I have code to generate this info in LCALS -- just need to
  //       pull it out and massage it into what we want)
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
