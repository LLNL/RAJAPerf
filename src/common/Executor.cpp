//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hpp"

#include "common/KernelBase.hpp"
#include "common/OutputUtils.hpp"

#ifdef RAJA_PERFSUITE_ENABLE_MPI
#include <mpi.h>
#endif

// Warmup kernels to run first to help reduce startup overheads in timings
#include "basic/DAXPY.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/INDEXLIST_3LOOP.hpp"
#include "algorithm/SORT.hpp"
#include "apps/HALOEXCHANGE_FUSED.hpp"

#include <list>
#include <vector>
#include <string>
#include <unordered_map>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <unistd.h>


namespace rajaperf {

using namespace std;

namespace {

#ifdef RAJA_PERFSUITE_ENABLE_MPI

void Allreduce(const Checksum_type* send, Checksum_type* recv, int count,
               MPI_Op op, MPI_Comm comm)
{
  if (op != MPI_SUM && op != MPI_MIN && op != MPI_MAX) {
    getCout() << "\nUnsupported MPI_OP..." << endl;
  }

  if (Checksum_MPI_type == MPI_DATATYPE_NULL) {

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    int num_ranks = -1;
    MPI_Comm_size(comm, &num_ranks);

    std::vector<Checksum_type> gather(count*num_ranks);

    MPI_Gather(send, count*sizeof(Checksum_type), MPI_BYTE,
               gather.data(), count*sizeof(Checksum_type), MPI_BYTE,
               0, comm);

    if (rank == 0) {

      for (int i = 0; i < count; ++i) {

        Checksum_type val = gather[i];

        for (int r = 1; r < num_ranks; ++r) {
          if (op == MPI_SUM) {
            val += gather[i + r*count];
          } else if (op == MPI_MIN) {
            val = std::min(val, gather[i + r*count]);
          } else if (op == MPI_MAX) {
            val = std::max(val, gather[i + r*count]);
          }
        }
        recv[i] = val;
      }

    }

    MPI_Bcast(recv, count*sizeof(Checksum_type), MPI_BYTE,
              0, comm);

  } else {

    MPI_Allreduce(send, recv, count, Checksum_MPI_type, op, comm);
  }

}

#endif

}

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    reference_vid(NumVariants),
    reference_tune_idx(KernelBase::getUnknownTuningIdx())
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

  getCout() << "\nSetting up suite based on input..." << endl;

  using Slist = list<string>;
  using Svector = vector<string>;
  using COvector = vector<RunParams::CombinerOpt>;
  using KIDset = set<KernelID>;
  using VIDset = set<VariantID>;

  //
  // Determine which kernels to exclude from input.
  // exclude_kern will be non-duplicated ordered set of IDs of kernel to exclude.
  //
  const Svector& npasses_combiner_input = run_params.getNpassesCombinerOptInput();
  if ( !npasses_combiner_input.empty() ) {

    COvector combiners;
    Svector invalid;
    for (const std::string& combiner_name : npasses_combiner_input) {

      if (combiner_name == RunParams::CombinerOptToStr(RunParams::CombinerOpt::Average)) {
        combiners.emplace_back(RunParams::CombinerOpt::Average);
      } else if (combiner_name == RunParams::CombinerOptToStr(RunParams::CombinerOpt::Minimum)) {
        combiners.emplace_back(RunParams::CombinerOpt::Minimum);
      } else if (combiner_name == RunParams::CombinerOptToStr(RunParams::CombinerOpt::Maximum)) {
        combiners.emplace_back(RunParams::CombinerOpt::Maximum);
      } else {
        invalid.emplace_back(combiner_name);
      }

    }

    run_params.setNpassesCombinerOpts(combiners);
    run_params.setInvalidNpassesCombinerOptInput(invalid);

  }

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
          reference_tune_idx = 0;
        }
      }
    }

    //
    // Set reference variant if not specified.
    //
    if ( run_params.getReferenceVariant().empty() && !run_var.empty() ) {
      reference_vid = *run_var.begin();
      reference_tune_idx = 0;
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
              reference_tune_idx = 0;
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
      reference_tune_idx = 0;
    }

    run_params.setInvalidVariantInput(invalid);

  }

  //
  // Create kernel objects and variants to execute. If invalid input is not
  // empty for either case, then there were unmatched input items.
  //
  // A message will be emitted later so user can sort it out...
  //

  if ( !(run_params.getInvalidNpassesCombinerOptInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput);

  } else if ( !(run_params.getInvalidKernelInput().empty()) ||
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
      }

      //
      // Make a single ordering of tuning names for each variant across kernels.
      //
      for (VariantID vid : variant_ids) {
        std::unordered_map<std::string, size_t> tuning_names_order_map;
        for (const KernelBase* kernel : kernels) {
          for (std::string const& tuning_name :
               kernel->getVariantTuningNames(vid)) {
            if (tuning_names_order_map.find(tuning_name) ==
                tuning_names_order_map.end()) {
              tuning_names_order_map.emplace(
                  tuning_name, tuning_names_order_map.size());
            }
          }
        }
        tuning_names[vid].resize(tuning_names_order_map.size());
        for (auto const& tuning_name_idx_pair : tuning_names_order_map) {
          tuning_names[vid][tuning_name_idx_pair.second] = tuning_name_idx_pair.first;
        }
        // reorder to put "default" first
        auto default_order_iter = tuning_names_order_map.find(KernelBase::getDefaultTuningName());
        if (default_order_iter != tuning_names_order_map.end()) {
          size_t default_idx = default_order_iter->second;
          std::string default_name = std::move(tuning_names[vid][default_idx]);
          tuning_names[vid].erase(tuning_names[vid].begin()+default_idx);
          tuning_names[vid].emplace(tuning_names[vid].begin(), std::move(default_name));
        }
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

    str << "\nVariants and Tunings"
        << "\n--------\n";
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      for (std::string const& tuning_name : tuning_names[variant_ids[iv]]) {
        str << getVariantName(variant_ids[iv]) << "-" << tuning_name<< endl;
      }
    }

    str << endl;

    bool to_file = false;
    writeKernelInfoSummary(str, to_file);

  }

  str.flush();
}


void Executor::writeKernelInfoSummary(ostream& str, bool to_file) const
{
  if ( to_file ) {
#ifdef RAJA_PERFSUITE_ENABLE_MPI
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    str << "Kernels run on " << num_ranks << " MPI ranks" << endl;
#else
    str << "Kernels run without MPI" << endl;
#endif
  }

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
      << sepchr <<right<< setw(flopsrep_width) << flopsrep_head
      << endl;

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

  getCout() << "\n\nRun warmup kernels...\n";

  vector<KernelBase*> warmup_kernels;

  warmup_kernels.push_back(makeKernel<basic::DAXPY>());
  warmup_kernels.push_back(makeKernel<basic::REDUCE3_INT>());
  warmup_kernels.push_back(makeKernel<basic::INDEXLIST_3LOOP>());
  warmup_kernels.push_back(makeKernel<algorithm::SORT>());
  warmup_kernels.push_back(makeKernel<apps::HALOEXCHANGE_FUSED>());

  for (size_t ik = 0; ik < warmup_kernels.size(); ++ik) {
    KernelBase* warmup_kernel = warmup_kernels[ik];
    runKernel(warmup_kernel, true);
    delete warmup_kernel;
    warmup_kernels[ik] = nullptr;
  }


  getCout() << "\n\nRunning specified kernels and variants...\n";

  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {
    if ( run_params.showProgress() ) {
      getCout() << "\nPass through suite # " << ip << "\n";
    }

    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      runKernel(kernel, false);
    } // loop over kernels

  } // loop over passes through suite

}

template < typename Kernel >
KernelBase* Executor::makeKernel()
{
  Kernel* kernel = new Kernel(run_params);
  return kernel;
}

void Executor::runKernel(KernelBase* kernel, bool print_kernel_name)
{
  if ( run_params.showProgress() || print_kernel_name) {
    getCout()  << endl << "Run kernel -- " << kernel->getName() << endl;
  }
  for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
    VariantID vid = variant_ids[iv];

    if ( run_params.showProgress() ) {
      if ( kernel->hasVariantDefined(vid) ) {
        getCout() << "   Running ";
      } else {
        getCout() << "   No ";
      }
      getCout() << getVariantName(vid) << " variant" << endl;
    }

    for (size_t tune_idx = 0; tune_idx < kernel->getNumVariantTunings(vid); ++tune_idx) {

      if ( run_params.showProgress() ) {
        getCout() << "     Running "
                  << kernel->getVariantTuningName(vid, tune_idx) << " tuning";
      }
      kernel->execute(vid, tune_idx);
      if ( run_params.showProgress() ) {
        getCout() << " -- " << kernel->getLastTime() << " sec." << endl;
      }
    }
  } // loop over variants
}

void Executor::outputRunData()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::PerfRun &&
       in_state != RunParams::CheckRun ) {
    return;
  }

  getCout() << "\n\nGenerate run report files...\n";

  //
  // Generate output file prefix (including directory path).
  //
  string out_fprefix;
  string outdir = recursiveMkdir(run_params.getOutputDirName());
  if ( !outdir.empty() ) {
    chdir(outdir.c_str());
  }
  out_fprefix = "./" + run_params.getOutputFilePrefix();

  unique_ptr<ostream> file;


  for (RunParams::CombinerOpt combiner : run_params.getNpassesCombinerOpts()) {
    file = openOutputFile(out_fprefix + "-timing-" + RunParams::CombinerOptToStr(combiner) + ".csv");
    writeCSVReport(*file, CSVRepMode::Timing, combiner, 6 /* prec */);

    if ( haveReferenceVariant() ) {
      file = openOutputFile(out_fprefix + "-speedup-" + RunParams::CombinerOptToStr(combiner) + ".csv");
      writeCSVReport(*file, CSVRepMode::Speedup, combiner, 3 /* prec */);
    }
  }

  file = openOutputFile(out_fprefix + "-checksum.txt");
  writeChecksumReport(*file);

  {
    vector<FOMGroup> fom_groups;
    getFOMGroups(fom_groups);
    if (!fom_groups.empty() ) {
      file = openOutputFile(out_fprefix + "-fom.csv");
      writeFOMReport(*file, fom_groups);
    }
  }

  file = openOutputFile(out_fprefix + "-kernels.csv");
  if ( *file ) {
    bool to_file = true;
    writeKernelInfoSummary(*file, to_file);
  }
}

unique_ptr<ostream> Executor::openOutputFile(const string& filename) const
{
  int rank = 0;
#ifdef RAJA_PERFSUITE_ENABLE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank == 0) {
    unique_ptr<ostream> file(new ofstream(filename.c_str(), ios::out | ios::trunc));
    if ( !*file ) {
      getCout() << " ERROR: Can't open output file " << filename << endl;
    }
    return file;
  }
  return unique_ptr<ostream>(makeNullStream());
}

void Executor::writeCSVReport(ostream& file, CSVRepMode mode,
                              RunParams::CombinerOpt combiner, size_t prec)
{
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

    vector<std::vector<size_t>> vartuncol_width(variant_ids.size());
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      size_t var_width = max(prec+2, getVariantName(variant_ids[iv]).size());
      for (std::string const& tuning_name : tuning_names[variant_ids[iv]]) {
        vartuncol_width[iv].emplace_back(max(var_width, tuning_name.size()));
      }
    }

    //
    // Print title line.
    //
    file << getReportTitle(mode, combiner);

    //
    // Wrtie CSV file contents for report.
    //

    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      for (size_t it = 0; it < tuning_names[variant_ids[iv]].size(); ++it) {
        file << sepchr;
      }
    }
    file << endl;

    //
    // Print column variant name line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      for (size_t it = 0; it < tuning_names[variant_ids[iv]].size(); ++it) {
        file << sepchr <<left<< setw(vartuncol_width[iv][it])
             << getVariantName(variant_ids[iv]);
      }
    }
    file << endl;

    //
    // Print column tuning name line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      for (size_t it = 0; it < tuning_names[variant_ids[iv]].size(); ++it) {
        file << sepchr <<left<< setw(vartuncol_width[iv][it])
             << tuning_names[variant_ids[iv]][it];
      }
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
        for (size_t it = 0; it < tuning_names[variant_ids[iv]].size(); ++it) {
          std::string const& tuning_name = tuning_names[variant_ids[iv]][it];
          file << sepchr <<right<< setw(vartuncol_width[iv][it]);
          if ( (mode == CSVRepMode::Speedup) &&
               (!kern->hasVariantTuningDefined(reference_vid, reference_tune_idx) ||
                !kern->hasVariantTuningDefined(vid, tuning_name)) ) {
            file << "Not run";
          } else if ( (mode == CSVRepMode::Timing) &&
                      !kern->hasVariantTuningDefined(vid, tuning_name) ) {
            file << "Not run";
          } else {
            file << setprecision(prec) << std::fixed
                 << getReportDataEntry(mode, combiner, kern, vid,
                        kern->getVariantTuningIndex(vid, tuning_name));
          }
        }
      }
      file << endl;
    }

    file.flush();

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeFOMReport(ostream& file, vector<FOMGroup>& fom_groups)
{
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

    std::vector<size_t> fom_group_ncols(fom_groups.size(), 0);
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];

      for (size_t gv = 0; gv < group.variants.size(); ++gv) {
        VariantID vid = group.variants[gv];
        const string& variant_name = getVariantName(vid);
        // num variants and tuning
        // Includes the PM baseline and the variants and tunings to compared to it
        fom_group_ncols[ifg] += tuning_names[vid].size();
        for (const string& tuning_name : tuning_names[vid]) {
          fom_col_width = max(fom_col_width, variant_name.size()+1+tuning_name.size());
        }
      }
    }

    vector< vector<int> > col_exec_count(fom_groups.size());
    vector< vector<double> > col_min(fom_groups.size());
    vector< vector<double> > col_max(fom_groups.size());
    vector< vector<double> > col_avg(fom_groups.size());
    vector< vector<double> > col_stddev(fom_groups.size());
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      col_exec_count[ifg].resize(fom_group_ncols[ifg], 0);
      col_min[ifg].resize(fom_group_ncols[ifg], numeric_limits<double>::max());
      col_max[ifg].resize(fom_group_ncols[ifg], -numeric_limits<double>::max());
      col_avg[ifg].resize(fom_group_ncols[ifg], 0.0);
      col_stddev[ifg].resize(fom_group_ncols[ifg], 0.0);
    }
    vector< vector< vector<double> > > pct_diff(kernels.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      pct_diff[ik].resize(fom_groups.size());
      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        pct_diff[ik][ifg].resize(fom_group_ncols[ifg], 0.0);
      }
    }

    //
    // Print title line.
    //
    file << "FOM Report : signed speedup(-)/slowdown(+) for each PM (base vs. RAJA) -> (T_RAJA - T_base) / T_base )";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t iv = 0; iv < fom_group_ncols[ifg]*2; ++iv) {
        file << sepchr;
      }
    }
    file << endl;

    file << "'OVER_TOL' in column to right if RAJA speedup is over tolerance";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t iv = 0; iv < fom_group_ncols[ifg]*2; ++iv) {
        file << sepchr;
      }
    }
    file << endl;

    string pass(",        ");
    string fail(",OVER_TOL");
    string base(",base_ref");

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      for (size_t gv = 0; gv < group.variants.size(); ++gv) {
        VariantID vid = group.variants[gv];
        string variant_name = getVariantName(vid);
        for (const string& tuning_name : tuning_names[vid]) {
          file << sepchr <<left<< setw(fom_col_width)
               << (variant_name+"-"+tuning_name) << pass;
        }
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

      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        constexpr double unknown_totTime = -1.0;
        double base_totTime = unknown_totTime;

        size_t col = 0;
        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID vid = group.variants[gv];

          for (const string& tuning_name : tuning_names[vid]) {

            size_t tune_idx = kern->getVariantTuningIndex(vid, tuning_name);

            //
            // If kernel variant was run, generate data for it and
            // print (signed) percentage difference from baseline.
            //
            if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
              col_exec_count[ifg][col]++;

              bool is_base = (base_totTime == unknown_totTime);
              if (is_base) {
                base_totTime = kern->getTotTime(vid, tune_idx);
              }

              pct_diff[ik][ifg][col] =
                (kern->getTotTime(vid, tune_idx) - base_totTime) / base_totTime;

              string pfstring(pass);
              if (pct_diff[ik][ifg][col] > run_params.getPFTolerance()) {
                pfstring = fail;
              }
              if (is_base) {
                pfstring = base;
              }

              file << sepchr << setw(fom_col_width) << setprecision(prec)
                   <<left<< pct_diff[ik][ifg][col] <<right<< pfstring;

              //
              // Gather data for column summaries (unsigned).
              //
              col_min[ifg][col] = min( col_min[ifg][col], pct_diff[ik][ifg][col] );
              col_max[ifg][col] = max( col_max[ifg][col], pct_diff[ik][ifg][col] );
              col_avg[ifg][col] += pct_diff[ik][ifg][col];

            } else {  // variant was not run, print a big fat goose egg...

              file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
                   << 0.0 << pass;

            }

            col++;
          }

        }  // loop over group variants

      }  // loop over fom_groups (i.e., columns)

      file << endl;

    } // loop over kernels


    //
    // Compute column summary data.
    //

    // Column average...
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          col_avg[ifg][col] /= col_exec_count[ifg][col];
        } else {
          col_avg[ifg][col] = 0.0;
        }
      }
    }

    // Column standard deviation...
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];

      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        int col = 0;
        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID vid = group.variants[gv];

          for (const string& tuning_name : tuning_names[vid]) {

            size_t tune_idx = kern->getVariantTuningIndex(vid, tuning_name);

            if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
              col_stddev[ifg][col] += ( pct_diff[ik][ifg][col] - col_avg[ifg][col] ) *
                                      ( pct_diff[ik][ifg][col] - col_avg[ifg][col] );
            }

            col++;
          }

        } // loop over group variants

      }  // loop over groups

    }  // loop over kernels

    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          col_stddev[ifg][col] /= col_exec_count[ifg][col];
        } else {
          col_stddev[ifg][col] = 0.0;
        }
      }
    }

    //
    // Print column summaries.
    //
    file <<left<< setw(kercol_width) << " ";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr << setw(fom_col_width) <<left<< "  " <<right<< pass;
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Min";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
             << col_min[ifg][col] << pass;
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Max";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
             << col_max[ifg][col] << pass;
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Avg";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
             << col_avg[ifg][col] << pass;
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Std Dev";
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
             << col_stddev[ifg][col] << pass;
      }
    }
    file << endl;

    file.flush();

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeChecksumReport(ostream& file)
{
  if ( file ) {

#ifdef RAJA_PERFSUITE_ENABLE_MPI
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
#endif

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
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t var_width = getVariantName(variant_ids[iv]).size();
        for (std::string const& tuning_name :
             kernels[ik]->getVariantTuningNames(variant_ids[iv])) {
          namecol_width = max(namecol_width, var_width+1+tuning_name.size());
        }
      }
    }
    namecol_width++;


    //
    // Print title.
    //
    file << equal_line << endl;
    file << "Checksum Report ";
#ifdef RAJA_PERFSUITE_ENABLE_MPI
    file << "for " << num_ranks << " MPI ranks ";
#endif
    file << endl;
    file << equal_line << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(namecol_width) << "Kernel  " << endl;
    file << dot_line << endl;
    file <<left<< setw(namecol_width) << "Variants  "
#ifdef RAJA_PERFSUITE_ENABLE_MPI
         <<left<< setw(checksum_width) << "Average Checksum  "
         <<left<< setw(checksum_width) << "Max Checksum Diff  "
         <<left<< setw(checksum_width) << "Checksum Diff StdDev"
#else
         <<left<< setw(checksum_width) << "Checksum  "
         <<left<< setw(checksum_width) << "Checksum Diff  "
#endif
         << endl;
    file <<left<< setw(namecol_width) << "  "
         <<left<< setw(checksum_width) << "  "
         <<left<< setw(checksum_width) << "(vs. first variant listed)  "
#ifdef RAJA_PERFSUITE_ENABLE_MPI
         <<left<< setw(checksum_width) << ""
#endif
         << endl;
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
        size_t num_tunings = kern->getNumVariantTunings(vid);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
            cksum_ref = kern->getChecksum(vid, tune_idx);
            found_ref = true;
            break;
          }
        }
        ++ivck;
      }

      // get vector of checksums and diffs
      std::vector<std::vector<Checksum_type>> checksums(variant_ids.size());
      std::vector<std::vector<Checksum_type>> checksums_diff(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);

        checksums[iv].resize(num_tunings, 0.0);
        checksums_diff[iv].resize(num_tunings, 0.0);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
            checksums[iv][tune_idx] = kern->getChecksum(vid, tune_idx);
            checksums_diff[iv][tune_idx] = cksum_ref - kern->getChecksum(vid, tune_idx);
          }
        }
      }

#ifdef RAJA_PERFSUITE_ENABLE_MPI

      // get stats for checksums
      std::vector<std::vector<Checksum_type>> checksums_sum(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_sum[iv].resize(num_tunings, 0.0);
        Allreduce(checksums[iv].data(), checksums_sum[iv].data(), num_tunings,
                  MPI_SUM, MPI_COMM_WORLD);
      }

      std::vector<std::vector<Checksum_type>> checksums_avg(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_avg[iv].resize(num_tunings, 0.0);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          checksums_avg[iv][tune_idx] = checksums_sum[iv][tune_idx] / num_ranks;
        }
      }

      // get stats for checksums_abs_diff
      std::vector<std::vector<Checksum_type>> checksums_abs_diff(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_abs_diff[iv].resize(num_tunings, 0.0);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          checksums_abs_diff[iv][tune_idx] = std::abs(checksums_diff[iv][tune_idx]);
        }
      }

      std::vector<std::vector<Checksum_type>> checksums_abs_diff_min(variant_ids.size());
      std::vector<std::vector<Checksum_type>> checksums_abs_diff_max(variant_ids.size());
      std::vector<std::vector<Checksum_type>> checksums_abs_diff_sum(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_abs_diff_min[iv].resize(num_tunings, 0.0);
        checksums_abs_diff_max[iv].resize(num_tunings, 0.0);
        checksums_abs_diff_sum[iv].resize(num_tunings, 0.0);

        Allreduce(checksums_abs_diff[iv].data(), checksums_abs_diff_min[iv].data(), num_tunings,
                  MPI_MIN, MPI_COMM_WORLD);
        Allreduce(checksums_abs_diff[iv].data(), checksums_abs_diff_max[iv].data(), num_tunings,
                  MPI_MAX, MPI_COMM_WORLD);
        Allreduce(checksums_abs_diff[iv].data(), checksums_abs_diff_sum[iv].data(), num_tunings,
                  MPI_SUM, MPI_COMM_WORLD);
      }

      std::vector<std::vector<Checksum_type>> checksums_abs_diff_avg(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_abs_diff_avg[iv].resize(num_tunings, 0.0);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          checksums_abs_diff_avg[iv][tune_idx] = checksums_abs_diff_sum[iv][tune_idx] / num_ranks;
        }
      }

      std::vector<std::vector<Checksum_type>> checksums_abs_diff_diff2avg2(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_abs_diff_diff2avg2[iv].resize(num_tunings, 0.0);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          checksums_abs_diff_diff2avg2[iv][tune_idx] = (checksums_abs_diff[iv][tune_idx] - checksums_abs_diff_avg[iv][tune_idx]) *
                                                  (checksums_abs_diff[iv][tune_idx] - checksums_abs_diff_avg[iv][tune_idx]) ;
        }
      }

      std::vector<std::vector<Checksum_type>> checksums_abs_diff_stddev(variant_ids.size());
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        checksums_abs_diff_stddev[iv].resize(num_tunings, 0.0);
        Allreduce(checksums_abs_diff_diff2avg2[iv].data(), checksums_abs_diff_stddev[iv].data(), num_tunings,
                  MPI_SUM, MPI_COMM_WORLD);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          checksums_abs_diff_stddev[iv][tune_idx] = std::sqrt(checksums_abs_diff_stddev[iv][tune_idx] / num_ranks);
        }
      }

#endif

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
        const string& variant_name = getVariantName(vid);

        size_t num_tunings = kernels[ik]->getNumVariantTunings(variant_ids[iv]);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          const string& tuning_name = kern->getVariantTuningName(vid, tune_idx);

          if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
            file <<left<< setw(namecol_width) << (variant_name+"-"+tuning_name)
                 << showpoint << setprecision(prec)
#ifdef RAJA_PERFSUITE_ENABLE_MPI
                 <<left<< setw(checksum_width) << checksums_avg[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_abs_diff_max[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_abs_diff_stddev[iv][tune_idx] << endl;
#else
                 <<left<< setw(checksum_width) << checksums[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_diff[iv][tune_idx] << endl;
#endif
          } else {
            file <<left<< setw(namecol_width) << (variant_name+"-"+tuning_name)
#ifdef RAJA_PERFSUITE_ENABLE_MPI
                 <<left<< setw(checksum_width) << "Not Run"
                 <<left<< setw(checksum_width) << "Not Run"
                 <<left<< setw(checksum_width) << "Not Run" << endl;
#else
                 <<left<< setw(checksum_width) << "Not Run"
                 <<left<< setw(checksum_width) << "Not Run" << endl;
#endif
          }

        }
      }

      file << endl;
      file << dash_line_short << endl;
    }

    file.flush();

  } // note file will be closed when file stream goes out of scope
}


string Executor::getReportTitle(CSVRepMode mode, RunParams::CombinerOpt combiner)
{
  string title;
  switch ( combiner ) {
    case RunParams::CombinerOpt::Average : {
      title = string("Mean ");
    }
    break;
    case RunParams::CombinerOpt::Minimum : {
      title = string("Min ");
    }
    break;
    case RunParams::CombinerOpt::Maximum : {
      title = string("Max ");
    }
    break;
    default : { getCout() << "\n Unknown CSV combiner mode = " << combiner << endl; }
  }
  switch ( mode ) {
    case CSVRepMode::Timing : {
      title += string("Runtime Report (sec.) ");
      break;
    }
    case CSVRepMode::Speedup : {
      if ( haveReferenceVariant() ) {
        title += string("Speedup Report (T_ref/T_var)") +
                 string(": ref var = ") + getVariantName(reference_vid) +
                 string(" ");
      }
      break;
    }
    default : { getCout() << "\n Unknown CSV report mode = " << mode << endl; }
  };
  return title;
}

long double Executor::getReportDataEntry(CSVRepMode mode,
                                         RunParams::CombinerOpt combiner,
                                         KernelBase* kern,
                                         VariantID vid,
                                         size_t tune_idx)
{
  long double retval = 0.0;
  switch ( mode ) {
    case CSVRepMode::Timing : {
      switch ( combiner ) {
        case RunParams::CombinerOpt::Average : {
          retval = kern->getTotTime(vid, tune_idx) / run_params.getNumPasses();
        }
        break;
        case RunParams::CombinerOpt::Minimum : {
          retval = kern->getMinTime(vid, tune_idx);
        }
        break;
        case RunParams::CombinerOpt::Maximum : {
          retval = kern->getMaxTime(vid, tune_idx);
        }
        break;
        default : { getCout() << "\n Unknown CSV combiner mode = " << combiner << endl; }
      }
      break;
    }
    case CSVRepMode::Speedup : {
      if ( haveReferenceVariant() ) {
        if ( kern->hasVariantTuningDefined(reference_vid, reference_tune_idx) &&
             kern->hasVariantTuningDefined(vid, tune_idx) ) {
          switch ( combiner ) {
            case RunParams::CombinerOpt::Average : {
              retval = kern->getTotTime(reference_vid, reference_tune_idx) /
                       kern->getTotTime(vid, tune_idx);
            }
            break;
            case RunParams::CombinerOpt::Minimum : {
              retval = kern->getMinTime(reference_vid, reference_tune_idx) /
                       kern->getMinTime(vid, tune_idx);
            }
            break;
            case RunParams::CombinerOpt::Maximum : {
              retval = kern->getMaxTime(reference_vid, reference_tune_idx) /
                       kern->getMaxTime(vid, tune_idx);
            }
            break;
            default : { getCout() << "\n Unknown CSV combiner mode = " << combiner << endl; }
          }
        } else {
          retval = 0.0;
        }
#if 0 // RDH DEBUG  (leave this here, it's useful for debugging!)
        getCout() << "Kernel(iv): " << kern->getName() << "(" << vid << ")"
                                                       << "(" << tune_idx << ")"endl;
        getCout() << "\tref_time, tot_time, retval = "
             << kern->getTotTime(reference_vid, reference_tune_idx) << " , "
             << kern->getTotTime(vid, tune_idx) << " , "
             << retval << endl;
#endif
      }
      break;
    }
    default : { getCout() << "\n Unknown CSV report mode = " << mode << endl; }
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
      group.variants.push_back(vid);

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
  getCout() << "\nFOMGroups..." << endl;
  for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
    const FOMGroup& group = fom_groups[ifg];
    getCout() << "\tBase : " << getVariantName(group.base) << endl;
    for (size_t iv = 0; iv < group.variants.size(); ++iv) {
      getCout() << "\t\t " << getVariantName(group.variants[iv]) << endl;
    }
  }
#endif
}



}  // closing brace for rajaperf namespace
