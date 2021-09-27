//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hpp"

#include "common/KernelBase.hpp"
#include "common/OutputUtils.hpp"

// Warmup kernels to run first to help reduce startup overheads in timings
#include "basic/DAXPY.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "algorithm/SORT.hpp"

#include <list>
#include <vector>
#include <string>
#include <regex>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

#include <unistd.h>

namespace rajaperf {

using namespace std;

vector<string> split(const string str, const string regex_str)
{
    regex regexz(regex_str);
    vector<string> list(sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  sregex_token_iterator());
    return list;
}

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    reference_vid(NumVariants)
{
#ifdef RAJAPERF_USE_CALIPER
  struct configuration cc;
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("perfsuite_version", cc.perfsuite_version);
  adiak::value("raja_version", cc.raja_version);
  adiak::value("cmake_build_type", cc.cmake_build_type);

  adiak::value("compiler_path", cc.compiler);
  cout << "Compiler path: " << cc.compiler << "\n";
  auto tokens = split(cc.compiler, "/");
  string compiler_exec = tokens.back();
  adiak::value("compiler_version", cc.compiler_version);
  string compiler = compiler_exec + "-" + cc.compiler_version;
  cout << "Compiler: " << compiler << "\n";
  adiak::value("compiler", compiler.c_str());
  auto tsize = tokens.size();
  if (tsize >= 3) {
    // pickup path version <compiler-version-hash|date>/bin/exec
    string path_version = tokens[tsize-3];
    cout << "Compiler path version: " << path_version << "\n";
    auto s = split(path_version,"-");
    if (s.size() >= 2) {
      string path_version_short = s[0] + "-" + s[1];
      cout << "Compiler path version short: " << path_version_short << "\n";
      adiak::value("Compiler_path_version",path_version_short.c_str());
    } 
  }


  adiak::value("compiler_flags", cc.compiler_flags);
  if (!strcmp(cc.cmake_build_type, "Release"))
    adiak::value("compiler_flags_release", cc.compiler_flags_release);
  else if (!strcmp(cc.cmake_build_type, "RelWithDebInfo"))
    adiak::value("compiler_flags_relwithdebinfo", cc.compiler_flags_relwithdebinfo);
  else if (!strcmp(cc.cmake_build_type, "Debug"))
    adiak::value("compiler_flags_debug", cc.compiler_flags_debug);

  if (strlen(cc.cuda_compiler_version) > 0) {
    adiak::value("cuda_compiler_version", cc.cuda_compiler_version);
    adiak::value("cuda_flags", cc.cuda_flags);
    adiak::value("cuda_flags_release", cc.cuda_flags_release);
  }

  if (strlen(cc.systype_build) > 0)
    adiak::value("systype_build", cc.systype_build);
  if (strlen(cc.machine_build) > 0)
    adiak::value("machine_build", cc.machine_build);

  adiak::value("ProblemSize",1.0);
  adiak::value("SizeMeaning",run_params.SizeMeaningToStr(run_params.getSizeMeaning()).c_str());
  if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Factor) {
    adiak::value("ProblemSize",run_params.getSizeFactor());
  } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
    adiak::value("ProblemSize",run_params.getSize());
  }

#endif
}


Executor::~Executor()
{
  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    delete kernels[ik];
  }
#ifdef RAJAPERF_USE_CALIPER 
  adiak::fini();
#endif
}


void Executor::setupSuite()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state == RunParams::InfoRequest || in_state == RunParams::BadInput ) {
    return;
  }

  cout << "\nSetting up suite based on input..." << endl;

  using Slist = list<string>;
  using Svector = vector<string>;
  using KIDset = set<KernelID>;
  using VIDset = set<VariantID>;

  //
  // Determine which kernels to exclude from input.
  // exclude_kern will be non-duplicated ordered set of IDs of kernel to exclude.
  //
  const Svector& exclude_kernel_input = run_params.getExcludeKernelInput();
  const Svector& exclude_feature_input = run_params.getExcludeFeatureInput();

  KIDset exclude_kern;

  if ( !exclude_kernel_input.empty() ) {

    // Make list copy of exclude kernel name input to manipulate for
    // processing potential group names and/or kernel names, next
    Slist exclude_kern_names(exclude_kernel_input.begin(), exclude_kernel_input.end());

    //
    // Search exclude_kern_names for matching group names.
    // groups2exclude will contain names of groups to exclude.
    //
    Svector groups2exclude;
    for (Slist::iterator it = exclude_kern_names.begin(); it != exclude_kern_names.end(); ++it)
    {
      for (size_t ig = 0; ig < NumGroups; ++ig) {
        const string& group_name = getGroupName(static_cast<GroupID>(ig));
        if ( group_name == *it ) {
          groups2exclude.push_back(group_name);
        }
      }
    }

    //
    // If group name(s) found in exclude_kern_names, assemble kernels in group(s)
    // to run and remove those group name(s) from exclude_kern_names list.
    //
    for (size_t ig = 0; ig < groups2exclude.size(); ++ig) {
      const string& gname(groups2exclude[ig]);

      for (size_t ik = 0; ik < NumKernels; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getFullKernelName(kid).find(gname) != string::npos ) {
          exclude_kern.insert(kid);
        }
      }

      exclude_kern_names.remove(gname);
    }

    //
    // Look for matching names of individual kernels in remaining exclude_kern_names.
    //
    // Assemble invalid input for warning message.
    //
    Svector invalid;

    for (Slist::iterator it = exclude_kern_names.begin(); it != exclude_kern_names.end(); ++it)
    {
      bool found_it = false;

      for (size_t ik = 0; ik < NumKernels && !found_it; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        if ( getKernelName(kid) == *it || getFullKernelName(kid) == *it ) {
          exclude_kern.insert(kid);
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(*it);
    }

    run_params.setInvalidExcludeKernelInput(invalid);

  }

  if ( !exclude_feature_input.empty() ) {

    // First, check for invalid exclude_feature input.
    // Assemble invalid input for warning message.
    //
    Svector invalid;

    for (size_t i = 0; i < exclude_feature_input.size(); ++i) {
      bool found_it = false;

      for (size_t fid = 0; fid < NumFeatures && !found_it; ++fid) {
        FeatureID tfid = static_cast<FeatureID>(fid);
        if ( getFeatureName(tfid) == exclude_feature_input[i] ) {
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back( exclude_feature_input[i] );
    }
    run_params.setInvalidExcludeFeatureInput(invalid);

    //
    // If feature input is valid, determine which kernels use
    // input-specified features and add to set of kernels to run.
    //
    if ( run_params.getInvalidExcludeFeatureInput().empty() ) {

      for (size_t i = 0; i < exclude_feature_input.size(); ++i) {

        const string& feature = exclude_feature_input[i];

        bool found_it = false;
        for (size_t fid = 0; fid < NumFeatures && !found_it; ++fid) {
          FeatureID tfid = static_cast<FeatureID>(fid);
          if ( getFeatureName(tfid) == feature ) {
            found_it = true;

            for (int kid = 0; kid < NumKernels; ++kid) {
              KernelID tkid = static_cast<KernelID>(kid);
              KernelBase* kern = getKernelObject(tkid, run_params);
              if ( kern->usesFeature(tfid) ) {
                 exclude_kern.insert( tkid );
              }
              delete kern;
            }  // loop over kernels

          }  // if input feature name matches feature id
        }  // loop over feature ids until name match is found

      }  // loop over feature name input

    }  // if feature name input is valid
  }

  //
  // Determine which kernels to execute from input.
  // run_kern will be non-duplicated ordered set of IDs of kernel to run.
  //
  const Svector& kernel_input = run_params.getKernelInput();
  const Svector& feature_input = run_params.getFeatureInput();

  KIDset run_kern;

  if ( kernel_input.empty() && feature_input.empty() ) {

    //
    // No kernels or features specified in input, run them all...
    //
    for (size_t kid = 0; kid < NumKernels; ++kid) {
      KernelID tkid = static_cast<KernelID>(kid);
      if (exclude_kern.find(tkid) == exclude_kern.end()) {
        run_kern.insert( tkid );
      }
    }

  } else {

    //
    // Need to parse input to determine which kernels to run
    //

    //
    // Look for kernels using features if such input provided
    //
    if ( !feature_input.empty() ) {

      // First, check for invalid feature input.
      // Assemble invalid input for warning message.
      //
      Svector invalid;

      for (size_t i = 0; i < feature_input.size(); ++i) {
        bool found_it = false;

        for (size_t fid = 0; fid < NumFeatures && !found_it; ++fid) {
          FeatureID tfid = static_cast<FeatureID>(fid);
          if ( getFeatureName(tfid) == feature_input[i] ) {
            found_it = true;
          }
        }

        if ( !found_it )  invalid.push_back( feature_input[i] );
      }
      run_params.setInvalidFeatureInput(invalid);

      //
      // If feature input is valid, determine which kernels use
      // input-specified features and add to set of kernels to run.
      //
      if ( run_params.getInvalidFeatureInput().empty() ) {

        for (size_t i = 0; i < feature_input.size(); ++i) {

          const string& feature = feature_input[i];

          bool found_it = false;
          for (size_t fid = 0; fid < NumFeatures && !found_it; ++fid) {
            FeatureID tfid = static_cast<FeatureID>(fid);
            if ( getFeatureName(tfid) == feature ) {
              found_it = true;

              for (int kid = 0; kid < NumKernels; ++kid) {
                KernelID tkid = static_cast<KernelID>(kid);
                KernelBase* kern = getKernelObject(tkid, run_params);
                if ( kern->usesFeature(tfid) &&
                     exclude_kern.find(tkid) == exclude_kern.end() ) {
                   run_kern.insert( tkid );
                }
                delete kern;
              }  // loop over kernels

            }  // if input feature name matches feature id
          }  // loop over feature ids until name match is found

        }  // loop over feature name input

      }  // if feature name input is valid

    } // if !feature_input.empty()

    // Make list copy of kernel name input to manipulate for
    // processing potential group names and/or kernel names, next
    Slist kern_names(kernel_input.begin(), kernel_input.end());

    //
    // Search kern_names for matching group names.
    // groups2run will contain names of groups to run.
    //
    Svector groups2run;
    for (Slist::iterator it = kern_names.begin(); it != kern_names.end(); ++it)
    {
      for (size_t ig = 0; ig < NumGroups; ++ig) {
        const string& group_name = getGroupName(static_cast<GroupID>(ig));
        if ( group_name == *it ) {
          groups2run.push_back(group_name);
        }
      }
    }

    //
    // If group name(s) found in kern_names, assemble kernels in group(s)
    // to run and remove those group name(s) from kern_names list.
    //
    for (size_t ig = 0; ig < groups2run.size(); ++ig) {
      const string& gname(groups2run[ig]);

      for (size_t kid = 0; kid < NumKernels; ++kid) {
        KernelID tkid = static_cast<KernelID>(kid);
        if ( getFullKernelName(tkid).find(gname) != string::npos &&
             exclude_kern.find(tkid) == exclude_kern.end()) {
          run_kern.insert(tkid);
        }
      }

      kern_names.remove(gname);
    }

    //
    // Look for matching names of individual kernels in remaining kern_names.
    //
    // Assemble invalid input for warning message.
    //
    Svector invalid;

    for (Slist::iterator it = kern_names.begin(); it != kern_names.end(); ++it)
    {
      bool found_it = false;

      for (size_t kid = 0; kid < NumKernels && !found_it; ++kid) {
        KernelID tkid = static_cast<KernelID>(kid);
        if ( getKernelName(tkid) == *it || getFullKernelName(tkid) == *it ) {
          if (exclude_kern.find(tkid) == exclude_kern.end()) {
            run_kern.insert(tkid);
          }
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(*it);
    }

    run_params.setInvalidKernelInput(invalid);

  }


  //
  // Assemble set of available variants to run
  // (based on compile-time configuration).
  //
  VIDset available_var;
  for (size_t iv = 0; iv < NumVariants; ++iv) {
    VariantID vid = static_cast<VariantID>(iv);
    if ( isVariantAvailable( vid ) ) {
       available_var.insert( vid );
    }
  }


  //
  // Determine variants to execute from input.
  // run_var will be non-duplicated ordered set of IDs of variants to run.
  //
  const Svector& exclude_variant_names = run_params.getExcludeVariantInput();

  VIDset exclude_var;

  if ( !exclude_variant_names.empty() ) {

    //
    // Parse input to determine which variants to exclude.
    //
    // Assemble invalid input for warning message.
    //

    Svector invalid;

    for (size_t it = 0; it < exclude_variant_names.size(); ++it) {
      bool found_it = false;

      for (VIDset::iterator vid_it = available_var.begin();
         vid_it != available_var.end(); ++vid_it) {
        VariantID vid = *vid_it;
        if ( getVariantName(vid) == exclude_variant_names[it] ) {
          exclude_var.insert(vid);
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(exclude_variant_names[it]);
    }

    run_params.setInvalidExcludeVariantInput(invalid);

  }

  //
  // Determine variants to execute from input.
  // run_var will be non-duplicated ordered set of IDs of variants to run.
  //
  const Svector& variant_names = run_params.getVariantInput();

  VIDset run_var;

  if ( variant_names.empty() ) {

    //
    // No variants specified in input options, run all available.
    // Also, set reference variant if specified.
    //
    for (VIDset::iterator vid_it = available_var.begin();
         vid_it != available_var.end(); ++vid_it) {
      VariantID vid = *vid_it;
      if (exclude_var.find(vid) == exclude_var.end()) {
        run_var.insert( vid );
        if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
          reference_vid = vid;
        }
      }
    }

    //
    // Set reference variant if not specified.
    //
    if ( run_params.getReferenceVariant().empty() && !run_var.empty() ) {
      reference_vid = *run_var.begin();
    }

  } else {

    //
    // Parse input to determine which variants to run:
    //   - variants to run will be the intersection of available variants
    //     and those specified in input
    //   - reference variant will be set to specified input if available
    //     and variant will be run; else first variant that will be run.
    //
    // Assemble invalid input for warning message.
    //

    Svector invalid;

    for (size_t it = 0; it < variant_names.size(); ++it) {
      bool found_it = false;

      for (VIDset::iterator vid_it = available_var.begin();
         vid_it != available_var.end(); ++vid_it) {
        VariantID vid = *vid_it;
        if ( getVariantName(vid) == variant_names[it] ) {
          if (exclude_var.find(vid) == exclude_var.end()) {
            run_var.insert(vid);
            if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
              reference_vid = vid;
            }
          }
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(variant_names[it]);
    }

    //
    // Set reference variant if not specified.
    //
    if ( run_params.getReferenceVariant().empty() && !run_var.empty() ) {
      reference_vid = *run_var.begin();
    }

    run_params.setInvalidVariantInput(invalid);

  }

  //
  // Create kernel objects and variants to execute. If invalid input is not
  // empty for either case, then there were unmatched input items.
  //
  // A message will be emitted later so user can sort it out...
  //

  if ( !(run_params.getInvalidKernelInput().empty()) ||
       !(run_params.getInvalidExcludeKernelInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput);

  } else if ( !(run_params.getInvalidFeatureInput().empty()) ||
              !(run_params.getInvalidExcludeFeatureInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput);

  } else { // kernel and feature input looks good

    for (KIDset::iterator kid = run_kern.begin();
         kid != run_kern.end(); ++kid) {
///   RDH DISABLE COUPLE KERNEL until we find a reasonable way to do
///   complex numbers in GPU code
      if ( *kid != Apps_COUPLE ) {
        kernels.push_back( getKernelObject(*kid, run_params) );
      }
    }

    if ( !(run_params.getInvalidVariantInput().empty()) ||
         !(run_params.getInvalidExcludeVariantInput().empty()) ) {

       run_params.setInputState(RunParams::BadInput);

    } else { // variant input lools good

      for (VIDset::iterator vid = run_var.begin();
           vid != run_var.end(); ++vid) {
        variant_ids.push_back( *vid );
#ifdef RAJAPERF_USE_CALIPER
        KernelBase::setCaliperMgrVariant(*vid);
#endif
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
    if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Factor) {
      str << "\t Kernel size factor = " << run_params.getSizeFactor() << endl;
    } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
      str << "\t Kernel size = " << run_params.getSize() << endl;
    }
    str << "\t Kernel rep factor = " << run_params.getRepFactor() << endl;
    str << "\t Output files will be named " << ofiles << endl;

    str << "\nThe following kernels and variants (when available for a kernel) will be run:" << endl;

    str << "\nVariants"
        << "\n--------\n";
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      str << getVariantName(variant_ids[iv]) << endl;
    }

    str << endl;

    bool to_file = false;
    writeKernelInfoSummary(str, to_file);

  }

  str.flush();
}


void Executor::writeKernelInfoSummary(ostream& str, bool to_file) const
{

//
// Set up column headers and column widths for kernel summary output.
//
  string kern_head("Kernels");
  size_t kercol_width = kern_head.size();

  Index_type psize_width = 0;
  Index_type reps_width = 0;
  Index_type itsrep_width = 0;
  Index_type bytesrep_width = 0;
  Index_type flopsrep_width = 0;
  Index_type dash_width = 0;

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    kercol_width = max(kercol_width, kernels[ik]->getName().size());
    psize_width = max(psize_width, kernels[ik]->getActualProblemSize());
    reps_width = max(reps_width, kernels[ik]->getRunReps());
    itsrep_width = max(reps_width, kernels[ik]->getItsPerRep());
    bytesrep_width = max(bytesrep_width, kernels[ik]->getBytesPerRep());
    flopsrep_width = max(bytesrep_width, kernels[ik]->getFLOPsPerRep());
  }

  const string sepchr(" , ");

  kercol_width += 2;
  dash_width += kercol_width;

  double psize = log10( static_cast<double>(psize_width) );
  string psize_head("Problem size");
  psize_width = max( static_cast<Index_type>(psize_head.size()),
                     static_cast<Index_type>(psize) ) + 3;
  dash_width += psize_width + static_cast<Index_type>(sepchr.size());

  double rsize = log10( static_cast<double>(reps_width) );
  string rsize_head("Reps");
  reps_width = max( static_cast<Index_type>(rsize_head.size()),
                    static_cast<Index_type>(rsize) ) + 3;
  dash_width += reps_width + static_cast<Index_type>(sepchr.size());

  double irsize = log10( static_cast<double>(itsrep_width) );
  string itsrep_head("Iterations/rep");
  itsrep_width = max( static_cast<Index_type>(itsrep_head.size()),
                      static_cast<Index_type>(irsize) ) + 3;
  dash_width += itsrep_width + static_cast<Index_type>(sepchr.size());

  string kernsrep_head("Kernels/rep");
  Index_type kernsrep_width =
    max( static_cast<Index_type>(kernsrep_head.size()),
         static_cast<Index_type>(4) );
  dash_width += kernsrep_width + static_cast<Index_type>(sepchr.size());

  double brsize = log10( static_cast<double>(bytesrep_width) );
  string bytesrep_head("Bytes/rep");
  bytesrep_width = max( static_cast<Index_type>(bytesrep_head.size()),
                        static_cast<Index_type>(brsize) ) + 3;
  dash_width += bytesrep_width + static_cast<Index_type>(sepchr.size());

  double frsize = log10( static_cast<double>(flopsrep_width) );
  string flopsrep_head("FLOPS/rep");
  flopsrep_width = max( static_cast<Index_type>(flopsrep_head.size()),
                         static_cast<Index_type>(frsize) ) + 3;
  dash_width += flopsrep_width + static_cast<Index_type>(sepchr.size());

  str <<left<< setw(kercol_width) << kern_head
      << sepchr <<right<< setw(psize_width) << psize_head
      << sepchr <<right<< setw(reps_width) << rsize_head
      << sepchr <<right<< setw(itsrep_width) << itsrep_head
      << sepchr <<right<< setw(kernsrep_width) << kernsrep_head
      << sepchr <<right<< setw(bytesrep_width) << bytesrep_head
      << sepchr <<right<< setw(flopsrep_width) << flopsrep_head << endl;

  if ( !to_file ) {
    for (Index_type i = 0; i < dash_width; ++i) {
      str << "-";
    }
    str << endl;
  }

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    KernelBase* kern = kernels[ik];
    str <<left<< setw(kercol_width) <<  kern->getName()
        << sepchr <<right<< setw(psize_width) << kern->getActualProblemSize()
        << sepchr <<right<< setw(reps_width) << kern->getRunReps()
        << sepchr <<right<< setw(itsrep_width) << kern->getItsPerRep()
        << sepchr <<right<< setw(kernsrep_width) << kern->getKernelsPerRep()
        << sepchr <<right<< setw(bytesrep_width) << kern->getBytesPerRep()
        << sepchr <<right<< setw(flopsrep_width) << kern->getFLOPsPerRep()
        << endl;
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

  cout << "\n\nRun warmup kernels...\n";

  vector<KernelBase*> warmup_kernels;

  warmup_kernels.push_back(new basic::DAXPY(run_params));
  warmup_kernels.push_back(new basic::REDUCE3_INT(run_params));
  warmup_kernels.push_back(new algorithm::SORT(run_params));

  for (size_t ik = 0; ik < warmup_kernels.size(); ++ik) {
    KernelBase* warmup_kernel = warmup_kernels[ik];
    cout << "Kernel : " << warmup_kernel->getName() << endl;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      VariantID vid = variant_ids[iv];
      if ( run_params.showProgress() ) {
        if ( warmup_kernel->hasVariantDefined(vid) ) {
          cout << "   Running ";
        } else {
          cout << "   No ";
        }
        cout << getVariantName(vid) << " variant" << endl;
      }
      if ( warmup_kernel->hasVariantDefined(vid) ) {
#ifdef RAJAPERF_USE_CALIPER
        warmup_kernel->caliperOff();
#endif
        warmup_kernel->execute(vid);
#ifdef RAJAPERF_USE_CALIPER
        warmup_kernel->caliperOn();
#endif
      }
      cout << getVariantName(vid) << " variant" << endl;
    }

    delete warmup_kernels[ik];
  }


  cout << "\n\nRunning specified kernels and variants...\n";

  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {
    if ( run_params.showProgress() ) {
      std::cout << "\nPass through suite # " << ip << "\n";
    }

    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      if ( run_params.showProgress() ) {
        std::cout << "\nRun kernel -- " << kernel->getName() << "\n";
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
         VariantID vid = variant_ids[iv];
         KernelBase* kern = kernels[ik];
         if ( run_params.showProgress() ) {
           if ( kern->hasVariantDefined(vid) ) {
             cout << "   Running ";
           } else {
             cout << "   No ";
           }
           cout << getVariantName(vid) << " variant" << endl;
         }
         if ( kern->hasVariantDefined(vid) ) {
           kernels[ik]->execute(vid);
         }
      } // loop over variants

    } // loop over kernels

  } // loop over passes through suite
#ifdef RAJAPERF_USE_CALIPER
  // Flush Caliper data
  KernelBase::setCaliperMgrFlush();
#endif
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

  filename = out_fprefix + "-kernels.csv";
  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {
    bool to_file = true;
    writeKernelInfoSummary(file, to_file);
  }
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
        file << sepchr <<right<< setw(varcol_width[iv]);
        if ( (mode == CSVRepMode::Speedup) &&
             (!kern->hasVariantDefined(reference_vid) ||
              !kern->hasVariantDefined(vid)) ) {
          file << "Not run";
        } else if ( (mode == CSVRepMode::Timing) &&
                    !kern->hasVariantDefined(vid) ) {
          file << "Not run";
        } else {
          file << setprecision(prec) << std::fixed
               << getReportDataEntry(mode, kern, vid);
        }
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
        } else {
          file <<left<< setw(namecol_width) << getVariantName(vid)
               <<left<< setw(checksum_width) << "Not Run"
               <<left<< setw(checksum_width) << "Not Run" << endl;
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
        if ( kern->hasVariantDefined(reference_vid) &&
             kern->hasVariantDefined(vid) ) {
          retval = kern->getTotTime(reference_vid) / kern->getTotTime(vid);
        } else {
          retval = 0.0;
        }
#if 0 // RDH DEBUG  (leave this here, it's useful for debugging!)
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

#if 0 //  RDH DEBUG   (leave this here, it's useful for debugging!)
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
