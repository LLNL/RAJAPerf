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

#include "common/KernelBase.hxx"
#include "common/OutputUtils.hxx"

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
  cout << "\nSetting up suite based on input..." << endl;

  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state == RunParams::InfoRequest || in_state == RunParams::BadInput ) {
    return;
  }

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
      reference_vid = VariantID::Baseline_Seq;
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

  cout << "\n\nRunning specified kernels and variants...\n";
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

  if ( haveReferenceVariant() ) { 
    filename = out_fprefix + "-speedup.csv";
    writeCSVReport(filename, CSVRepMode::Speedup);
  }

  filename = out_fprefix + "-checksum.txt";
  writeChecksumReport(filename);

  filename = out_fprefix + "-fom.csv";
  writeFOMReport(filename);
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
        file << sepchr <<left<< setw(varcol_width[iv]) << setprecision(prec) 
             << getReportDataEntry(mode, kern, vid);
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
    size_t prec = 16;

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()); 
    }
    kercol_width++;

    size_t fom_col_width = prec+8;

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
    vector< vector<double> > rel_diff(kernels.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      rel_diff[ik] = vector<double>(ncols, 0.0);
    }

    //
    // Print title line.
    //
    file << "FOM Report (relative timing diff: (T_RAJA - T_baseline) / T_baseline)";
    for (size_t iv = 0; iv < ncols; ++iv) {
      file << sepchr;
    }
    file << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      for (size_t gv = 0; gv < group.variants.size(); ++gv) {
        string name = getVariantName(group.variants[gv]);
        file << sepchr <<left<< setw(fom_col_width) << name; 
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

        VariantID base_vid = group.baseline;

        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID comp_vid = group.variants[gv];

          //
          // If kernel variant was run, generate data for it and
          // print (signed) percentage difference from baseline.
          // 
          if ( kern->wasVariantRun(comp_vid) ) {
            col_exec_count[col]++;

#if 0 // RDH RETHINK
            rel_diff[ik][col] = 
              (kern->getTotTime(comp_vid) - kern->getTotTime(base_vid)) /
               kern->getTotTime(base_vid);
#else
            rel_diff[ik][col] = 
              kern->getTotTime(comp_vid) / kern->getTotTime(base_vid);
#endif

            file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
                 << rel_diff[ik][col]; 

            //
            // Gather data for column summaries (unsigned).
            //  
            col_min[col] = min( col_min[col], rel_diff[ik][col] );
            col_max[col] = max( col_max[col], rel_diff[ik][col] );
            col_avg[col] += rel_diff[ik][col];

          } else {  // variant was not run, print a big fat goose egg...

            file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
                 << 0.0;

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
            col_stddev[col] += ( rel_diff[ik][col] - col_avg[col] ) *
                               ( rel_diff[ik][col] - col_avg[col] );
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
      file << sepchr <<left<< setw(fom_col_width) << " ";
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Min";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_min[col];
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Max";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_max[col];
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Avg";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_avg[col];
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Std Dev";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_stddev[col];
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
    const string dash_line("----------------------------------------------------------------------------------------------------");
    const string dash_line_short("-------------------------------------------------------");
    string dot_line("........................................................");

    size_t prec = 32;
    size_t checksum_width = prec + 4;

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
      while ( ivck < variant_ids.size() && cksum_ref == 0.0 ) {
        VariantID vid = variant_ids[ivck];
        if ( kern->wasVariantRun(vid) ) {
          cksum_ref = kern->getChecksum(vid);
        }
        ++ivck;
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
 
        Checksum_type diff = 0.0;    
        if ( kern->wasVariantRun(vid) ) {
          diff = cksum_ref - kern->getChecksum(vid);

          file <<left<< setw(namecol_width) << getVariantName(vid)
               << showpoint << setprecision(prec) 
               <<left<< setw(checksum_width) << cksum_ref
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
      title = string("Mean Runtime Report "); 
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

    if ( vname.find("Baseline") != string::npos ) {

      FOMGroup group;
      group.baseline = vid;
 
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

    }  // if variant name contains 'Baseline'

  }  // iterate over variant ids to run

#if 0 // RDH for debugging...
  cout << "\nFOMGroups..." << endl;
  for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
    const FOMGroup& group = fom_groups[ifg];
    cout << "\tBaseline : " << getVariantName(group.baseline) << endl;
    for (size_t iv = 0; iv < group.variants.size(); ++iv) {
      cout << "\t\t " << getVariantName(group.variants[iv]) << endl;
    }
  }
#endif
}



}  // closing brace for rajaperf namespace
