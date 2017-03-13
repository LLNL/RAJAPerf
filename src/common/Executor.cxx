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
#include "common/OutputUtils.hxx"

#include <list>
#include <set>
#include <vector>
#include <string>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <unistd.h>


namespace rajaperf {

using namespace std;

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    ref_vid(NumVariants)
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

  typedef list<string> Slist;
  typedef vector<string> Svector;
  typedef set<string> Sset;
  typedef set<KernelID> KIDset;
  typedef set<VariantID> VIDset;

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
        const string& group_name = getGroupName(static_cast<GroupID>(ig));
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
      const string& gname(groups2run[ig]);

      for (size_t ik = 0; ik < NumKernels; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(gname) != string::npos ) {
          run_kern.insert(kid);
        }
      }

      input.remove(gname);
    }

    //
    // Look for matching names of individual kernels in remaining input.
    // 
    // Assemble invalid input for warning message.
    //
    Svector invalid;

    for (Slist::iterator it = input.begin(); it != input.end(); ++it) {
      bool found_it = false;

      for (size_t ik = 0; ik < NumKernels && !found_it; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(*it) != string::npos ) {
          run_kern.insert(kid);
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(*it); 
    }

    run_params.setInvalidKernelInput(invalid);

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
    // Add reference variant to run variants if valid
    //
    for (size_t iv = 0; iv < NumVariants; ++iv) {
      VariantID vid = static_cast<VariantID>(iv);
      if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
        run_var.insert(vid);
        ref_vid = vid; 
      }
    }

    //
    // Need to parse input to determine which variants to run
    // 

    //
    // Search input for matching variant names.
    //
    // Assemble invalid input for warning message.
    //
    Svector invalid;

    for (size_t it = 0; it < variant_input.size(); ++it) {
      bool found_it = false;

      for (size_t iv = 0; iv < NumVariants && !found_it; ++iv) {
        VariantID vid = static_cast<VariantID>(iv);
        if ( getVariantName(vid) == variant_input[it] ) {
          run_var.insert(vid);
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(variant_input[it]);
    }

    run_params.setInvalidVariantInput(invalid);

  }

  //
  // Create kernel objects and variants to execute. If invalid input is not 
  // empty for either case, then there were unmatched input items.
  // 
  // A message will be emitted later so user can sort it out...
  //

  if ( !(run_params.getInvalidKernelInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput); 

  } else { // kernel input looks good

    for (KIDset::iterator kid = run_kern.begin(); 
         kid != run_kern.end(); ++kid) {
      kernels.push_back( getKernelObject(*kid, run_params) );
    }

    if ( !(run_params.getInvalidVariantInput().empty()) ) {

       run_params.setInputState(RunParams::BadInput);

    } else { // variant input lools good

      for (VIDset::iterator vid = run_var.begin();
           vid != run_var.end(); ++vid) {
        variant_ids.push_back( *vid );
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


void Executor::reportRunSummary(ostream& str) const
{
  RunParams::InputOpt in_state = run_params.getInputState();

  if ( in_state == RunParams::BadInput ) {

    str << "\nRunParams state:\n";
    str <<   "----------------";
    run_params.print(str);

    str << "\n\nSuite will not be run now due to bad input."
        << "\n  See run parameters or option messages above.\n" 
        << endl;

  } else if ( in_state == RunParams::GoodToRun || 
              in_state == RunParams::DryRun ) {

    if ( in_state == RunParams::DryRun ) {

      str << "\n\nRAJA performance suite dry run summary...."
          <<   "\n--------------------------------------" << endl;
 
      str << "\nRunParams state:";
      str << "\n----------------";
      run_params.print(str);

    } 

    if ( in_state == RunParams::GoodToRun ) {

      str << "\n\nRAJA performance suite run summary...."
          <<   "\n--------------------------------------" << endl;

    }

    string ofiles;
    if ( !run_params.getOutputDirName().empty() ) {
      ofiles = run_params.getOutputDirName();
    } else {
      ofiles = string(".");
    }
    ofiles += string("/") + run_params.getOutputFilePrefix() + 
              string("*");

    str << "\nSuite will run with kernels and variants listed below.\n" 
        << "Output files will be named " << ofiles << endl;

    str << "\nVariants"
        << "\n--------\n";
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      str << getVariantName(variant_ids[iv]) << endl;
    }

    str << "\nKernels"
        << "\n-------\n";
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      str << kernels[ik]->getName() << endl;
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

  cout << "\n\nRunning specified kernels and variants!\n";
  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {  
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
         kernels[ik]->execute( variant_ids[iv] );
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

  //
  // Generate output file prefix (including directory path). 
  //
  string out_fprefix;
  string outdir = recursiveMkdir(run_params.getOutputDirName()); 
  if ( !outdir.empty() ) {
    out_fprefix = outdir + "/" + run_params.getOutputFilePrefix();
    chdir(outdir.c_str());
  } else {
    out_fprefix = "./" + run_params.getOutputFilePrefix();
  }

  string filename = out_fprefix + "-timing.csv";
  writeCSVReport(filename, CSVRepMode::Timing);

  if ( ref_vid < NumVariants) { 
    filename = out_fprefix + "-speedup.csv";
    writeCSVReport(filename, CSVRepMode::Speedup);
  }

#if 0
  filename = out_fprefix + "-checksum.txt";
  writeChecksumReport(filename);

  filename = out_fprefix + "-fom.csv";
  writeReport(filename, CSVRepMode::FOM);
#endif
}


void Executor::writeCSVReport(const string& filename, CSVRepMode mode)
{
  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string kernel_col_name("Kernel  ");
    const string sepchr(" , ");
    size_t prec = 12;

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()+1); 
    }

    vector<size_t> varcol_width(variant_ids.size());
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      varcol_width[iv] = max(prec+2, getVariantName(variant_ids[iv]).size()); 
    } 

    //
    // Print title line.
    //
    file << getReportTitle(mode) << endl;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      file << sepchr;
    }
    file << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      file << sepchr <<left<< setw(varcol_width[iv])
           << getVariantName(variant_ids[iv]);
    }
    file << endl;

    //
    // Print row of variant run times for each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      file <<left<< setw(kercol_width) << kernels[ik]->getName();
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = static_cast<VariantID>(iv);
        file << sepchr <<left<< setw(varcol_width[iv]) << setprecision(prec) 
             << getReportDataEntry(mode, kernels[ik], vid);
      }
      file << endl;
    }

    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


string Executor::getReportTitle(CSVRepMode mode)
{
  string title;
  switch ( mode ) {
    case CSVRepMode::Timing : { 
      title = string("Mean Runtime Report "); 
      break; 
    }
    case CSVRepMode::Speedup : { 
      title = string("Speedup Report (T_ref/T_var)"); 
      if ( ref_vid < NumVariants ) {
        title = title + string(", ref var = ") +
                        getVariantName(ref_vid) + string(" ");
      }
      break; 
    }
    default : { cout << "\n Unknown CSV report mode = " << mode << endl; }
  }; 
  return title;
}

long double Executor::getReportDataEntry(CSVRepMode mode,
                                         KernelBase* kern, 
                                         VariantID vid)
{
  long double retval = 0.0; 
  switch ( mode ) {
    case CSVRepMode::Timing : { 
      retval = kern->getTotTime(vid) / run_params.getNumPasses();
      break; 
    }
    case CSVRepMode::Speedup : { 
      if ( ref_vid < NumVariants ) {
        long double var_time = 
          getReportDataEntry(CSVRepMode::Timing, kern, vid);
        long double ref_time = 
          getReportDataEntry(CSVRepMode::Timing, kern, ref_vid);
        retval = ref_time / var_time;
      }
      break; 
    }
    default : { cout << "\n Unknown CSV report mode = " << mode << endl; }
  }; 
  return retval;
}



}  // closing brace for rajaperf namespace
