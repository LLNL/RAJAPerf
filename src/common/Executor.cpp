//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hpp"

#include "common/KernelBase.hpp"
#include "common/OutputUtils.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include <mpi.h>
#endif

#include "CudaDataUtils.hpp"
#include "HipDataUtils.hpp"

// Warmup kernels to run first to help reduce startup overheads in timings
#include "basic/DAXPY.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/INDEXLIST_3LOOP.hpp"
#include "algorithm/SORT.hpp"
#include "apps/HALOEXCHANGE_FUSED.hpp"

#include <list>
#include <vector>
#include <string>
#include <regex>
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

#if defined(RAJA_PERFSUITE_USE_CALIPER)
vector<string> split(const string str, const string regex_str)
{
  regex regexz(regex_str);
  vector<string> list(sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                sregex_token_iterator());
  return list;
}
#endif

namespace {

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

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
#if defined(RAJA_PERFSUITE_USE_CALIPER)
  configuration cc;
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::value("perfsuite_version", cc.adiak_perfsuite_version);
  adiak::value("raja_version", cc.adiak_raja_version);
  adiak::value("cmake_build_type", cc.adiak_cmake_build_type);
  adiak::value("cmake_cxx_flags", cc.adiak_cmake_cxx_flags);
  adiak::value("rajaperf_compiler", cc.adiak_rajaperf_compiler);
  adiak::value("compiler_version", cc.adiak_compiler_version);

  auto tokens = split(cc.adiak_rajaperf_compiler, "/");
  string compiler_exec = tokens.back();
  adiak::catstring compiler = compiler_exec + "-" + std::string(cc.adiak_compiler_version);
  cout << "Compiler: " << (string)compiler << "\n";
  adiak::value("compiler", compiler);
  auto tsize = tokens.size();
  if (tsize >= 3) {
    // pickup path version <compiler-version-hash|date>/bin/exec
    string path_version = tokens[tsize-3];
    auto s = split(path_version,"-");
    if (s.size() >= 2) {
      adiak::path path_version_short = s[0] + "-" + s[1];
      adiak::value("Compiler_path_version", (adiak::catstring)path_version_short);
    } 
  }

  if (cc.adiak_cmake_exe_linker_flags.size() > 0) {
    adiak::value("cmake_exe_linker_flags", cc.adiak_cmake_exe_linker_flags);
  }
  if (cc.adiak_rajaperf_compiler_options.size() > 0) {
    adiak::value("rajaperf_compiler_options", cc.adiak_rajaperf_compiler_options);
  }
  if (std::string(cc.adiak_cuda_compiler_version).size() > 0) {
    adiak::value("cuda_compiler_version", cc.adiak_cuda_compiler_version);
  }
  if (strlen(cc.adiak_gpu_targets) > 0) {
    adiak::value("gpu_targets", cc.adiak_gpu_targets);
  }
  if (strlen(cc.adiak_cmake_hip_architectures) > 0) {
    adiak::value("cmake_hip_architectures", cc.adiak_cmake_hip_architectures);
  }
  if (cc.adiak_gpu_targets_block_sizes.size() > 0) {
    adiak::value("gpu_targets_block_sizes", cc.adiak_gpu_targets_block_sizes);
  }
  if (cc.adiak_raja_hipcc_flags.size() > 0) {
    adiak::value("raja_hipcc_flags", cc.adiak_raja_hipcc_flags);
  }
  if (std::string(cc.adiak_mpi_cxx_compiler).size() > 0) {
    adiak::value("mpi_cxx_compiler", cc.adiak_mpi_cxx_compiler);
  }
  if (std::string(cc.adiak_systype_build).size() > 0) {
    adiak::value("systype_build", cc.adiak_systype_build);
  }
  if (std::string(cc.adiak_machine_build).size() > 0) {
    adiak::value("machine_build", cc.adiak_machine_build);
  }

  adiak::value("SizeMeaning",(adiak::catstring)run_params.SizeMeaningToStr(run_params.getSizeMeaning()));
  if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Factor) {
    adiak::value("ProblemSizeRunParam",(uint)run_params.getSizeFactor());
  } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
    adiak::value("ProblemSizeRunParam",(uint)run_params.getSize());
  }

  // Openmp section
#if defined(_OPENMP)
  std::string strval = "";
  std::string test = std::to_string(_OPENMP);

  std::unordered_map<unsigned,std::string> map{
    {200505,"2.5"},{200805,"3.0"},{201107,"3.1"},{201307,"4.0"},{201511,"4.5"},{201611,"4.5"},{201811,"5.0"},{202011,"5.1"},{202111,"5.2"}};

  if (map.find(_OPENMP) != map.end()) {
    strval = map.at(_OPENMP);
  } else {
    strval="Version Not Detected";
  }

  std::cerr << "_OPENMP:" << test << " at version: " << strval << "\n";
  adiak::value("omp_version",(adiak::version)strval);
  uint ompthreads = omp_get_max_threads();
  adiak::value("omp_max_threads",ompthreads);
#endif

#endif
}


Executor::~Executor()
{
  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    delete kernels[ik];
  }
#if defined(RAJA_PERFSUITE_USE_CALIPER)
  adiak::fini();
#endif
}


void Executor::setupSuite()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state == RunParams::InfoRequest || in_state == RunParams::BadInput ) {
    return;
  }

  getCout() << "\nSetting up suite based on input..." << endl;

#if defined(RAJA_ENABLE_SYCL)
  KernelBase::qu = camp::resources::Sycl().get_queue();
#endif

  using Svector = vector<string>;

  //
  // Configure suite to run based on kernel and variant input.
  //
  // Kernel and variant input is assumed to be good at this point.
  //

  const std::set<KernelID>& run_kern = run_params.getKernelIDsToRun();
  for (auto kid = run_kern.begin(); kid != run_kern.end(); ++kid) {
    kernels.push_back( getKernelObject(*kid, run_params) );
  }

  const std::set<VariantID>& run_var = run_params.getVariantIDsToRun();
  for (auto vid = run_var.begin(); vid != run_var.end(); ++vid) {
    variant_ids.push_back( *vid );
  }

  //
  // Set reference variant and reference tuning index IDs. 
  //
  // This logic seems a bit strange. Not sure why we do it this way.
  //
  reference_vid = run_params.getReferenceVariantID();
  if ( reference_vid == NumVariants && !variant_ids.empty() ) {
    reference_tune_idx = 0;
  }
  for (auto vid = variant_ids.begin(); vid != variant_ids.end(); ++vid) {
    if ( *vid == reference_vid ) {
      reference_tune_idx = 0;
    }
  }


  //
  // Set up ordered list of tuning names based on kernels and variants
  // selected to run and tuning input.
  //
  // Member variable tuning_names will hold ordered list of tunings.
  //
  // Note that all tuning input has been checked for correctness at this point.
  //

  const Svector& selected_tuning_names = run_params.getTuningInput();
  const Svector& excluded_tuning_names = run_params.getExcludeTuningInput();

  for (VariantID vid : variant_ids) {

    std::unordered_map<std::string, size_t> tuning_names_order_map;
    for (const KernelBase* kernel : kernels) {

      for (std::string const& tuning_name :
          kernel->getVariantTuningNames(vid)) {

        if ( tuning_names_order_map.find(tuning_name) ==
             tuning_names_order_map.end()) {
          if ( (selected_tuning_names.empty() || 
                find(selected_tuning_names.begin(), 
                     selected_tuning_names.end(), tuning_name) != 
                selected_tuning_names.end() ) 
                  // If argument is not provided or name is selected
                  &&
             find(excluded_tuning_names.begin(), 
                  excluded_tuning_names.end(), tuning_name) == 
                  excluded_tuning_names.end()) { 
               // name does not exist in exclusion list
               tuning_names_order_map.emplace( tuning_name, 
                                               tuning_names_order_map.size()); 
             }  // find logic
        }  //  tuning_name is not in map 

      }  // iterate over kernel tuning variants

    }  // iterate over kernels

    tuning_names[vid].resize(tuning_names_order_map.size());
    for (auto const& tuning_name_idx_pair : tuning_names_order_map) {
      size_t const& tid = tuning_name_idx_pair.second;
      std::string const& tstr = tuning_name_idx_pair.first;
      tuning_names[vid][tid] = tstr;
#if defined(RAJA_PERFSUITE_USE_CALIPER)
      KernelBase::setCaliperMgrVariantTuning(vid,
                                             tstr,
                                             run_params.getOutputDirName(),
                                             run_params.getAddToSpotConfig());
#endif
    }

    // reorder to put "default" first
    auto default_order_iter = 
      tuning_names_order_map.find(KernelBase::getDefaultTuningName());
    if ( default_order_iter != tuning_names_order_map.end() ) {
      size_t default_idx = default_order_iter->second;
      std::string default_name = std::move(tuning_names[vid][default_idx]);
      tuning_names[vid].erase(tuning_names[vid].begin()+default_idx);
      tuning_names[vid].emplace(tuning_names[vid].begin(), 
                                std::move(default_name));
    }

  }  // iterate over variant_ids to run

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

    str << "\nData Spaces"
        << "\n--------";
    str << "\nSeq - " << getDataSpaceName(run_params.getSeqDataSpace());
    if (isVariantAvailable(VariantID::Base_OpenMP)) {
      str << "\nOpenMP - " << getDataSpaceName(run_params.getOmpDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_OpenMPTarget)) {
      str << "\nOpenMP Target - " << getDataSpaceName(run_params.getOmpTargetDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_CUDA)) {
      str << "\nCuda - " << getDataSpaceName(run_params.getCudaDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_HIP)) {
      str << "\nHip - " << getDataSpaceName(run_params.getHipDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_SYCL)) {
      str << "\nSycl - " << getDataSpaceName(run_params.getSyclDataSpace());
    }
    if (isVariantAvailable(VariantID::Kokkos_Lambda)) {
      str << "\nKokkos - " << getDataSpaceName(run_params.getKokkosDataSpace());
    }
    str << endl;

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
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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

  runWarmupKernels();

  getCout() << "\n\nRunning specified kernels and variants...\n";

  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {
    if ( run_params.showProgress() ) {
      getCout() << "\nPass through suite # " << ip << "\n";
    }

    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      runKernel(kernel, false);
    } // iterate over kernels

  } // iterate over passes through suite

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
        getCout() << "\tRunning ";
      } else {
        getCout() << "\tNo ";
      }
      getCout() << getVariantName(vid) << " variant" << endl;
    }

    for (size_t tune_idx = 0; 
         tune_idx < kernel->getNumVariantTunings(vid); 
         ++tune_idx) {
      std::string const& tuning_name = 
        kernel->getVariantTuningName(vid, tune_idx);

      if ( find(tuning_names[vid].begin(), 
                tuning_names[vid].end(), tuning_name) != 
             tuning_names[vid].end()) 
      { 
        // Check if valid tuning
        if ( run_params.showProgress() ) {
          getCout() << "\t\tRunning " << tuning_name << " tuning";
        }

        kernel->execute(vid, tune_idx); // Execute kernel

        if ( run_params.showProgress() ) {
          getCout() << " -- " << kernel->getLastTime() << " sec." << endl;
        }

      } else {
        getCout() << "\t\tSkipping " << tuning_name << " tuning" << endl;
      }

    }  // iterate over tunings 

  } // iterate over variants

}

void Executor::runWarmupKernels()
{
  if ( run_params.getDisableWarmup() ) {
    return;
  } 

  getCout() << "\n\nRun warmup kernels...\n";

  //
  // For kernels to be run, assemble a set of feature IDs
  //
  std::set<FeatureID> feature_ids;
  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    KernelBase* kernel = kernels[ik];

    for (size_t fid = 0; fid < NumFeatures; ++fid) {
      FeatureID tfid = static_cast<FeatureID>(fid);
      if (kernel->usesFeature(tfid) ) {
         feature_ids.insert( tfid );
      }
    }
  
  } // iterate over kernels

  //
  // Map feature IDs to set of warmup kernel IDs
  //
  std::set<KernelID> kernel_ids;
  for ( auto fid = feature_ids.begin(); fid != feature_ids.end(); ++ fid ) {

    switch (*fid) {

      case Forall:
      case Kernel:
      case Launch:
        kernel_ids.insert(Basic_DAXPY); break;

      case Sort:
        kernel_ids.insert(Algorithm_SORT); break;
   
      case Scan:
        kernel_ids.insert(Basic_INDEXLIST_3LOOP); break;

      case Workgroup:
        kernel_ids.insert(Apps_HALOEXCHANGE_FUSED); break;

      case Reduction:
        kernel_ids.insert(Basic_REDUCE3_INT); break;

      case Atomic:
        kernel_ids.insert(Basic_PI_ATOMIC); break; 

      case View:
        break;
  
      default:
        break;

    }

  }

  //
  // Run warmup kernels
  //
  for ( auto kid = kernel_ids.begin(); kid != kernel_ids.end(); ++ kid ) {
    KernelBase* kernel = getKernelObject(*kid, run_params);
#if defined(RAJA_PERFSUITE_USE_CALIPER)
    kernel->caliperOff();
#endif
    runKernel(kernel, true);
#if defined(RAJA_PERFSUITE_USE_CALIPER)
    kernel->caliperOn();
#endif
    delete kernel;
  }

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

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  KernelBase::setCaliperMgrFlush();
#endif
}

unique_ptr<ostream> Executor::openOutputFile(const string& filename) const
{
  int rank = 0;
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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

        }  // iterate over group variants

      }  // iterate over fom_groups (i.e., columns)

      file << endl;

    } // iterate over kernels


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

        } // iterate over group variants

      }  // iterate over groups

    }  // iterate over kernels

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

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

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
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
                 <<left<< setw(checksum_width) << checksums_avg[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_abs_diff_max[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_abs_diff_stddev[iv][tune_idx] << endl;
#else
                 <<left<< setw(checksum_width) << checksums[iv][tune_idx]
                 <<left<< setw(checksum_width) << checksums_diff[iv][tune_idx] << endl;
#endif
          } else {
            file <<left<< setw(namecol_width) << (variant_name+"-"+tuning_name)
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
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
