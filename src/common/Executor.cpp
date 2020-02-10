//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hpp"

#include "common/KernelBase.hpp"
#include "common/OutputUtils.hpp"

// Warmup kernel to run first to remove startup overheads in timings
#include "basic/DAXPY.hpp"

#include <list>
#include <vector>
#include <string>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

#include <unistd.h>


namespace rajaperf {

using namespace std;

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    reference_vid(NumVariants)
{
  cout << "\n\nReading command line input..." << endl;
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

  cout << "\nSetting up suite based on input..." << endl;

  typedef list<string> Slist;
  typedef vector<string> Svector;
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
        if ( getKernelName(kid) == *it || getFullKernelName(kid) == *it ) {
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
    // No variants specified in input options, run them all.
    // Also, set reference variant if specified.
    //
    for (size_t iv = 0; iv < NumVariants; ++iv) {
      VariantID vid = static_cast<VariantID>(iv);
      run_var.insert( vid );
      if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
        reference_vid = vid;
      }
    }

    //
    // Set reference variant if not specified.
    //
    if ( run_params.getReferenceVariant().empty() ) {
      reference_vid = VariantID::Base_Seq;
    }

  } else {

    //
    // Add reference variant to run variants if specified
    //
    for (size_t iv = 0; iv < NumVariants; ++iv) {
      VariantID vid = static_cast<VariantID>(iv);
      if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
        run_var.insert(vid);
        reference_vid = vid; 
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
/// RDH DISABLE COUPLE KERNEL until we find a reasonable way to do 
/// complex numbers in GPU code
      if ( *kid != Apps_COUPLE ) {
        kernels.push_back( getKernelObject(*kid, run_params) );
      }
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
      if ( run_params.getInputState() != RunParams::DryRun && 
           run_params.getInputState() != RunParams::CheckRun ) {
        run_params.setInputState(RunParams::PerfRun);
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

  } else if ( in_state == RunParams::PerfRun || 
              in_state == RunParams::DryRun || 
              in_state == RunParams::CheckRun ) {

    if ( in_state == RunParams::DryRun ) {

      str << "\n\nRAJA performance suite dry run summary...."
          <<   "\n--------------------------------------" << endl;
 
      str << "\nInput state:";
      str << "\n------------";
      run_params.print(str);

    } 

    if ( in_state == RunParams::PerfRun ||
         in_state == RunParams::CheckRun ) {

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

    str << "\nHow suite will be run:" << endl;
    str << "\t # passes = " << run_params.getNumPasses() << endl;
    str << "\t Kernel size factor = " << run_params.getSizeFactor() << endl;
    str << "\t Kernel rep factor = " << run_params.getRepFactor() << endl;
    str << "\t Output files will be named " << ofiles << endl;

    str << "\nThe following kernels and variants will be run:\n"; 

    str << "\nVariants"
        << "\n--------\n";
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      str << getVariantName(variant_ids[iv]) << endl;
    }

    str << "\nKernels(iterations/rep , reps)"
        << "\n-----------------------------\n";
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];
      str << kern->getName() 
          << " (" << kern->getItsPerRep() << " , "
          << kern->getRunReps() << ")" << endl;
    }

  }

  str.flush();
}

void Executor::runSuite()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::PerfRun && 
       in_state != RunParams::CheckRun ) {
    return;
  }

  cout << "\n\nRunning warmup kernel variants...\n";

  KernelBase* warmup_kernel = new basic::DAXPY(run_params);

  for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
    if ( run_params.showProgress() ) {
      cout << "Warmup Kernel " <<  getVariantName(variant_ids[iv]) << endl;
    }
    warmup_kernel->execute( variant_ids[iv] );
  }

  delete warmup_kernel;


  cout << "\n\nRunning specified kernels and variants...\n";

  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {
    if ( run_params.showProgress() ) {
      std::cout << "\nPass through suite # " << ip << "\n";
    }

    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      if ( run_params.showProgress() ) {
        std::cout << "\n   Running kernel -- " << kernel->getName() << "\n"; 
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
         KernelBase* kern = kernels[ik];
         if ( run_params.showProgress() ) {
           cout << kern->getName() << " " <<  getVariantName(variant_ids[iv]) << endl;
         }  
         kernels[ik]->execute( variant_ids[iv] );
      } // loop over variants 

    } // loop over kernels

  } // loop over passes through suite

}

void Executor::outputRunData()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::PerfRun && 
       in_state != RunParams::CheckRun ) {
    return;
  }

  cout << "\n\nGenerate run report files...\n";

  //
  // Generate output file prefix (including directory path). 
  //
  string out_fprefix;
  string outdir = recursiveMkdir(run_params.getOutputDirName()); 
  if ( !outdir.empty() ) {
    chdir(outdir.c_str());
  }
  out_fprefix = "./" + run_params.getOutputFilePrefix();

  string filename = out_fprefix + "-timing.csv";
  writeCSVReport(filename, CSVRepMode::Timing, 6 /* prec */);

  if ( haveReferenceVariant() ) { 
    filename = out_fprefix + "-speedup.csv";
    writeCSVReport(filename, CSVRepMode::Speedup, 3 /* prec */);
  }

  filename = out_fprefix + "-checksum.txt";
  writeChecksumReport(filename);

  filename = out_fprefix + "-fom.csv";
  writeFOMReport(filename);
}


void Executor::writeCSVReport(const string& filename, CSVRepMode mode,
                              size_t prec)
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

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()); 
    }
    kercol_width++;

    vector<size_t> varcol_width(variant_ids.size());
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      varcol_width[iv] = max(prec+2, getVariantName(variant_ids[iv]).size()); 
    } 

    //
    // Print title line.
    //
    file << getReportTitle(mode);

    //
    // Wrtie CSV file contents for report.
    // 

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
    // Print row of data for variants of each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];
      file <<left<< setw(kercol_width) << kern->getName();
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
        file << sepchr <<right<< setw(varcol_width[iv]) << setprecision(prec) 
             << std::fixed << getReportDataEntry(mode, kern, vid);
      }
      file << endl;
    }

    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeFOMReport(const string& filename)
{
  vector<FOMGroup> fom_groups; 
  getFOMGroups(fom_groups);
  if (fom_groups.empty() ) {
    return;
  }

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
    size_t prec = 2;

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()); 
    }
    kercol_width++;

    size_t fom_col_width = prec+14;

    size_t ncols = 0;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      ncols += group.variants.size(); // num variants to compare 
                                      // to each PM baseline
    }

    vector<int> col_exec_count(ncols, 0);
    vector<double> col_min(ncols, numeric_limits<double>::max());
    vector<double> col_max(ncols, -numeric_limits<double>::max());
    vector<double> col_avg(ncols, 0.0);
    vector<double> col_stddev(ncols, 0.0);
    vector< vector<double> > pct_diff(kernels.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      pct_diff[ik] = vector<double>(ncols, 0.0);
    }

    //
    // Print title line.
    //
    file << "FOM Report : signed speedup(-)/slowdown(+) for each PM (base vs. RAJA) -> (T_RAJA - T_base) / T_base )";
    for (size_t iv = 0; iv < ncols*2; ++iv) {
      file << sepchr;
    }
    file << endl;

    file << "'OVER_TOL' in column to right if RAJA speedup is over tolerance";
    for (size_t iv = 0; iv < ncols*2; ++iv) {
      file << sepchr;
    }
    file << endl;

    string pass(",        ");
    string fail(",OVER_TOL");

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      for (size_t gv = 0; gv < group.variants.size(); ++gv) {
        string name = getVariantName(group.variants[gv]);
        file << sepchr <<left<< setw(fom_col_width) << name << pass; 
      } 
    }
    file << endl;


    //
    // Write CSV file contents for FOM report.
    // 

    //
    // Print row of FOM data for each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];          

      file <<left<< setw(kercol_width) << kern->getName();
     
      int col = 0;
      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        VariantID base_vid = group.base;

        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID comp_vid = group.variants[gv];

          //
          // If kernel variant was run, generate data for it and
          // print (signed) percentage difference from baseline.
          // 
          if ( kern->wasVariantRun(comp_vid) ) {
            col_exec_count[col]++;

            pct_diff[ik][col] = 
              (kern->getTotTime(comp_vid) - kern->getTotTime(base_vid)) /
               kern->getTotTime(base_vid);

            string pfstring(pass);
            if (pct_diff[ik][col] > run_params.getPFTolerance()) {
              pfstring = fail;
            }

            file << sepchr << setw(fom_col_width) << setprecision(prec)
                 <<left<< pct_diff[ik][col] <<right<< pfstring;

            //
            // Gather data for column summaries (unsigned).
            //  
            col_min[col] = min( col_min[col], pct_diff[ik][col] );
            col_max[col] = max( col_max[col], pct_diff[ik][col] );
            col_avg[col] += pct_diff[ik][col];

          } else {  // variant was not run, print a big fat goose egg...

            file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
                 << 0.0 << pass;

          }

          col++;

        }  // loop over group variants

      }  // loop over fom_groups (i.e., columns)

      file << endl;

    } // loop over kernels


    // 
    // Compute column summary data.
    // 

    // Column average...
    for (size_t col = 0; col < ncols; ++col) {
      if ( col_exec_count[col] > 0 ) {
        col_avg[col] /= col_exec_count[col];
      } else {
        col_avg[col] = 0.0;
      }
    } 

    // Column standard deviaation...
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];          

      int col = 0;
      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID comp_vid = group.variants[gv];

          if ( kern->wasVariantRun(comp_vid) ) {
            col_stddev[col] += ( pct_diff[ik][col] - col_avg[col] ) *
                               ( pct_diff[ik][col] - col_avg[col] );
          } 

          col++;

        } // loop over group variants

      }  // loop over groups

    }  // loop over kernels
 
    for (size_t col = 0; col < ncols; ++col) {
      if ( col_exec_count[col] > 0 ) {
        col_stddev[col] /= col_exec_count[col];
      } else {
        col_stddev[col] = 0.0;
      }
    }

    // 
    // Print column summaries.
    // 
    file <<left<< setw(kercol_width) << " ";
    for (size_t iv = 0; iv < ncols; ++iv) {
      file << sepchr << setw(fom_col_width) <<left<< "  " <<right<< pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Min";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_min[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Max";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_max[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Avg";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_avg[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Std Dev";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_stddev[col] << pass;
    }
    file << endl;

    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeChecksumReport(const string& filename)
{
  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string equal_line("===================================================================================================");
    const string dash_line("----------------------------------------------------------------------------------------");
    const string dash_line_short("-------------------------------------------------------");
    string dot_line("........................................................");

    size_t prec = 20;
    size_t checksum_width = prec + 8;

    size_t namecol_width = 0;
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      namecol_width = max(namecol_width, kernels[ik]->getName().size()); 
    }
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      namecol_width = max(namecol_width, 
                          getVariantName(variant_ids[iv]).size()); 
    }
    namecol_width++;


    //
    // Print title.
    //
    file << equal_line << endl;
    file << "Checksum Report " << endl;
    file << equal_line << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(namecol_width) << "Kernel  " << endl;
    file << dot_line << endl;
    file <<left<< setw(namecol_width) << "Variants  " 
         <<left<< setw(checksum_width) << "Checksum  " 
         <<left<< setw(checksum_width) 
         << "Checksum Diff (vs. first variant listed)"; 
    file << endl;
    file << dash_line << endl;

    //
    // Print checksum and diff against baseline for each kernel variant.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];

      file <<left<< setw(namecol_width) << kern->getName() << endl;
      file << dot_line << endl;

      Checksum_type cksum_ref = 0.0;
      size_t ivck = 0;
      bool found_ref = false;
      while ( ivck < variant_ids.size() && !found_ref ) {
        VariantID vid = variant_ids[ivck];
        if ( kern->wasVariantRun(vid) ) {
          cksum_ref = kern->getChecksum(vid);
          found_ref = true;
        }
        ++ivck;
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
 
        if ( kern->wasVariantRun(vid) ) {
          Checksum_type vcheck_sum = kern->getChecksum(vid);
          Checksum_type diff = cksum_ref - kern->getChecksum(vid);

          file <<left<< setw(namecol_width) << getVariantName(vid)
               << showpoint << setprecision(prec) 
               <<left<< setw(checksum_width) << vcheck_sum
               <<left<< setw(checksum_width) << diff << endl;
        }
      }

      file << endl;
      file << dash_line_short << endl;
    }
    
    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


string Executor::getReportTitle(CSVRepMode mode)
{
  string title;
  switch ( mode ) {
    case CSVRepMode::Timing : { 
      title = string("Mean Runtime Report (sec.) "); 
      break; 
    }
    case CSVRepMode::Speedup : { 
      if ( haveReferenceVariant() ) {
        title = string("Speedup Report (T_ref/T_var)") +
                string(": ref var = ") + getVariantName(reference_vid) + 
                string(" ");
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
      if ( haveReferenceVariant() ) {
        retval = kern->getTotTime(reference_vid) / kern->getTotTime(vid);
#if 0 // RDH DEBUG
        cout << "Kernel(iv): " << kern->getName() << "(" << vid << ")" << endl;
        cout << "\tref_time, tot_time, retval = " 
             << kern->getTotTime(reference_vid) << " , "
             << kern->getTotTime(vid) << " , "
             << retval << endl;
#endif
      }
      break; 
    }
    default : { cout << "\n Unknown CSV report mode = " << mode << endl; }
  }; 
  return retval;
}

void Executor::getFOMGroups(vector<FOMGroup>& fom_groups)
{
  fom_groups.clear();

  for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
    VariantID vid = variant_ids[iv];
    string vname = getVariantName(vid);

    if ( vname.find("Base") != string::npos ) {

      FOMGroup group;
      group.base = vid;
 
      string::size_type pos = vname.find("_");
      string pm(vname.substr(pos+1, string::npos));

      for (size_t ivs = iv+1; ivs < variant_ids.size(); ++ivs) {
        VariantID vids = variant_ids[ivs];
        if ( getVariantName(vids).find(pm) != string::npos ) {
          group.variants.push_back(vids);
        }
      }

      if ( !group.variants.empty() ) {
        fom_groups.push_back( group );
      }

    }  // if variant name contains 'Base'

  }  // iterate over variant ids to run

#if 0 // RDH for debugging...
  cout << "\nFOMGroups..." << endl;
  for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
    const FOMGroup& group = fom_groups[ifg];
    cout << "\tBase : " << getVariantName(group.base) << endl;
    for (size_t iv = 0; iv < group.variants.size(); ++iv) {
      cout << "\t\t " << getVariantName(group.variants[iv]) << endl;
    }
  }
#endif
}



}  // closing brace for rajaperf namespace
