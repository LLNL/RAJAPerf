//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

// Warmup kernels for default warmup mode
#include "basic/DAXPY.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/INDEXLIST_3LOOP.hpp"
#include "algorithm/SORT.hpp"
#include "comm/HALO_PACKING_FUSED.hpp"
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include "comm/HALO_EXCHANGE_FUSED.hpp"
#endif

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

#if defined(_WIN32)
#include<direct.h>
#else
#include <unistd.h>
#endif


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

using active_variant_tunings_type = vector<pair<VariantID, vector<string>>>;

active_variant_tunings_type getActiveVariantTunings(
    vector<KernelBase*> const& kernels,
    vector<VariantID> const& variant_ids,
    const vector<string> (&tuning_names)[NumVariants])
{
  active_variant_tunings_type active_vartuns;
  for (VariantID vid : variant_ids) {
    pair<VariantID, vector<string>>* active_var = nullptr;
    for (string const& tuning_name : tuning_names[vid]) {
      for (KernelBase* kern : kernels) {
        if ( kern->wasVariantTuningRun(vid, tuning_name) ) {
          if (!active_var) {
            active_var = &active_vartuns.emplace_back(vid, 0);
          }
          auto& active_tuning_names = active_var->second;
          if (find(active_tuning_names.begin(), active_tuning_names.end(), tuning_name)
              == active_tuning_names.end()) {
            active_tuning_names.emplace_back(tuning_name);
          }
          break;
        }
      }
    }
  }
  return active_vartuns;
}

struct ChecksumData
{
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  std::vector<Checksum_type> checksums_avg;
  std::vector<Checksum_type> checksums_rel_diff_max;
  std::vector<Checksum_type> checksums_rel_diff_stddev;
#else
  std::vector<Checksum_type> checksums;
  std::vector<Checksum_type> checksums_rel_diff;
#endif
};

ChecksumData getChecksumData(KernelBase* kern, VariantID vid)
{
  size_t num_tunings = kern->getNumVariantTunings(vid);

  // get vector of checksums and diffs
  std::vector<Checksum_type> checksums(num_tunings, 0.0);
  std::vector<Checksum_type> checksums_rel_diff(num_tunings, 0.0);
  for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
    if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
      checksums[tune_idx] = kern->getChecksumAverage(vid, tune_idx);
      checksums_rel_diff[tune_idx] = kern->getChecksumMaxRelativeAbsoluteDifference(vid, tune_idx);
    }
  }

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // get stats for checksums
  std::vector<Checksum_type> checksums_sum(num_tunings, 0.0);
  Allreduce(checksums.data(), checksums_sum.data(), num_tunings,
            MPI_SUM, MPI_COMM_WORLD);

  std::vector<Checksum_type> checksums_avg(num_tunings, 0.0);
  for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
    checksums_avg[tune_idx] = checksums_sum[tune_idx] / num_ranks;
  }

  std::vector<Checksum_type> checksums_rel_diff_min(num_tunings, 0.0);
  std::vector<Checksum_type> checksums_rel_diff_max(num_tunings, 0.0);
  std::vector<Checksum_type> checksums_rel_diff_sum(num_tunings, 0.0);
  Allreduce(checksums_rel_diff.data(), checksums_rel_diff_min.data(), num_tunings,
            MPI_MIN, MPI_COMM_WORLD);
  Allreduce(checksums_rel_diff.data(), checksums_rel_diff_max.data(), num_tunings,
            MPI_MAX, MPI_COMM_WORLD);
  Allreduce(checksums_rel_diff.data(), checksums_rel_diff_sum.data(), num_tunings,
            MPI_SUM, MPI_COMM_WORLD);

  std::vector<Checksum_type> checksums_rel_diff_avg(num_tunings, 0.0);
  for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
    checksums_rel_diff_avg[tune_idx] = checksums_rel_diff_sum[tune_idx] / num_ranks;
  }

  std::vector<Checksum_type> checksums_rel_diff_diff2avg2(num_tunings, 0.0);
  for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
    checksums_rel_diff_diff2avg2[tune_idx] = (checksums_rel_diff[tune_idx] - checksums_rel_diff_avg[tune_idx]) *
                                             (checksums_rel_diff[tune_idx] - checksums_rel_diff_avg[tune_idx]) ;
  }

  std::vector<Checksum_type> checksums_rel_diff_stddev(num_tunings, 0.0);
  Allreduce(checksums_rel_diff_diff2avg2.data(), checksums_rel_diff_stddev.data(), num_tunings,
            MPI_SUM, MPI_COMM_WORLD);
  for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
    checksums_rel_diff_stddev[tune_idx] = std::sqrt(checksums_rel_diff_stddev[tune_idx] / num_ranks);
  }

#endif

  return ChecksumData {
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
        std::move(checksums_avg),
        std::move(checksums_rel_diff_max),
        std::move(checksums_rel_diff_stddev)
#else
        std::move(checksums),
        std::move(checksums_rel_diff)
#endif
      };
}

} // end unnamed namespace

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    reference_vid(NumVariants),
    reference_tune_idx(KernelBase::getUnknownTuningIdx())
{
#if defined(RAJA_PERFSUITE_USE_CALIPER)
  configuration cc;
  #if defined(RAJA_PERFSUITE_ENABLE_MPI)
    MPI_Comm adiak_comm = MPI_COMM_WORLD;
    adiak::init(&adiak_comm);
  #else
    adiak::init(nullptr);
  #endif
  adiak::collect_all();
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
  if (strlen(cc.adiak_tuning_cuda_arch) > 0) {
    adiak::value("tuning_cuda_arch", cc.adiak_tuning_cuda_arch);
  }
  if (strlen(cc.adiak_tuning_hip_arch) > 0) {
    adiak::value("tuning_hip_arch", cc.adiak_tuning_hip_arch);
  }
  if (cc.adiak_gpu_block_sizes.size() > 0) {
    adiak::value("gpu_block_sizes", cc.adiak_gpu_block_sizes);
  }
  if (cc.adiak_atomic_replications.size() > 0) {
    adiak::value("atomic_replications", cc.adiak_atomic_replications);
  }
  if (cc.adiak_gpu_items_per_thread.size() > 0) {
    adiak::value("gpu_items_per_thread", cc.adiak_gpu_items_per_thread);
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
  adiak::value("ProblemSizeRunParam",(uint)run_params.getSize());
  adiak::value("MemoryMeaning",(adiak::catstring)run_params.MemoryMeaningToStr(run_params.getMemoryMeaning()));
  adiak::value("MemorySizeRunParam",(uint)run_params.getMemory());
  adiak::value("ProblemSizeFactorRunParam",(uint)run_params.getSizeFactor());
  adiak::value("ProblemMinSizeRunParam",(uint)run_params.getMinSize());

  adiak::value("LtimesNumGroups", (uint)run_params.getLtimesNumG());
  adiak::value("LtimesNumMoments", (uint)run_params.getLtimesNumM());
  adiak::value("LtimesNumDirections", (uint)run_params.getLtimesNumD());

  adiak::value("FemsweepPolar", (uint)run_params.getFemsweepPolar());
  adiak::value("FemsweepAzim", (uint)run_params.getFemsweepAzim());
  adiak::value("FemsweepGroups", (uint)run_params.getFemsweepGroups());
  adiak::value("FemsweepX", (uint)run_params.getFemsweepX());
  adiak::value("FemsweepY", (uint)run_params.getFemsweepY());
  adiak::value("FemsweepZ", (uint)run_params.getFemsweepZ());

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
                                             run_params.getOutputFilePrefix(),
                                             run_params.getAddToSpotConfig(),
                                             run_params.getAddToCaliperConfig(),
                                             tuning_names_order_map.size());
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
    if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Default) {
      str << "\t Kernel size = Default" << endl;
    } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
      str << "\t Kernel size = " << run_params.getSize() << endl;
    } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Memory) {
      str << "\t Kernel memory " << run_params.MemoryMeaningToStr(run_params.getMemoryMeaning()) << " = " << run_params.getMemory() << endl;
    }
    str << "\t Kernel min size = " << run_params.getMinSize() << endl;
    str << "\t Kernel size factor = " << run_params.getSizeFactor() << endl;
    str << "\t Kernel rep factor = " << run_params.getRepFactor() << endl;
    str << "\t Output files will be named " << ofiles << endl;

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
    str << "\nRunning with " << run_params.getMPISize() << " MPI procs" << endl;
    auto div3d = run_params.getMPI3DDivision();
    const char* valid3d = run_params.validMPI3DDivision() ? "" : "invalid";
    str << "\t 3D division = " << div3d[0] << " x " << div3d[1] << " x " << div3d[2] << " " << valid3d << endl;
#endif

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

    str << "\nReduction Data Spaces"
        << "\n--------";
    str << "\nSeq - " << getDataSpaceName(run_params.getSeqReductionDataSpace());
    if (isVariantAvailable(VariantID::Base_OpenMP)) {
      str << "\nOpenMP - " << getDataSpaceName(run_params.getOmpReductionDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_OpenMPTarget)) {
      str << "\nOpenMP Target - " << getDataSpaceName(run_params.getOmpTargetReductionDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_CUDA)) {
      str << "\nCuda - " << getDataSpaceName(run_params.getCudaReductionDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_HIP)) {
      str << "\nHip - " << getDataSpaceName(run_params.getHipReductionDataSpace());
    }
    if (isVariantAvailable(VariantID::Kokkos_Lambda)) {
      str << "\nKokkos - " << getDataSpaceName(run_params.getKokkosReductionDataSpace());
    }
    str << endl;

    str << "\nMPI Data Spaces"
        << "\n--------";
    str << "\nSeq - " << getDataSpaceName(run_params.getSeqMPIDataSpace());
    if (isVariantAvailable(VariantID::Base_OpenMP)) {
      str << "\nOpenMP - " << getDataSpaceName(run_params.getOmpMPIDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_OpenMPTarget)) {
      str << "\nOpenMP Target - " << getDataSpaceName(run_params.getOmpTargetMPIDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_CUDA)) {
      str << "\nCuda - " << getDataSpaceName(run_params.getCudaMPIDataSpace());
    }
    if (isVariantAvailable(VariantID::Base_HIP)) {
      str << "\nHip - " << getDataSpaceName(run_params.getHipMPIDataSpace());
    }
    if (isVariantAvailable(VariantID::Kokkos_Lambda)) {
      str << "\nKokkos - " << getDataSpaceName(run_params.getKokkosMPIDataSpace());
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

    constexpr bool to_file = true;
    writeKernelInfoSummary(str, kernels, !to_file);

  }

  str.flush();
}


void Executor::writeKernelInfoSummary(ostream& str,
                                      vector<KernelBase*> const& kernels,
                                      bool to_file) const
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
  size_t     kernel_width = 0;
  Index_type psize_width = 0;
  Index_type reps_width = 0;
  Index_type itsrep_width = 0;
  Index_type kernsrep_width = 0;
  Index_type bytesMovedrep_width = 0;
  Index_type flopsrep_width = 0;
  Index_type bytesTouchedrep_width = 0;
  Index_type bytesReadrep_width = 0;
  Index_type bytesWrittenrep_width = 0;
  Index_type bytesModifyWrittenrep_width = 0;
  Index_type bytesAtomicModifyWrittenrep_width = 0;
  Index_type bytesAllocatedrep_width = 0;
  size_t     checksumConsistency_width = 0;
  size_t     operationalComplexity_width = 0;
  Index_type maxPerfectLoopDimensions_width = 0;
  Index_type problemDimensionality_width = 0;

  size_t     dash_width = 0;

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    kernel_width = max(kernel_width, kernels[ik]->getName().size());
    psize_width = max(psize_width, kernels[ik]->getActualProblemSize());
    reps_width = max(reps_width, kernels[ik]->getRunReps());
    itsrep_width = max(itsrep_width, kernels[ik]->getItsPerRep());
    kernsrep_width = max(kernsrep_width, kernels[ik]->getKernelsPerRep());
    bytesMovedrep_width = max(bytesMovedrep_width, kernels[ik]->getBytesMovedPerRep());
    flopsrep_width = max(flopsrep_width, kernels[ik]->getFLOPsPerRep());
    bytesTouchedrep_width = max(bytesTouchedrep_width, kernels[ik]->getBytesTouchedPerRep());
    bytesReadrep_width = max(bytesReadrep_width, kernels[ik]->getBytesReadPerRep());
    bytesWrittenrep_width = max(bytesWrittenrep_width, kernels[ik]->getBytesWrittenPerRep());
    bytesModifyWrittenrep_width = max(bytesModifyWrittenrep_width, kernels[ik]->getBytesModifyWrittenPerRep());
    bytesAtomicModifyWrittenrep_width = max(bytesAtomicModifyWrittenrep_width, kernels[ik]->getBytesAtomicModifyWrittenPerRep());
    bytesAllocatedrep_width = max(bytesAllocatedrep_width, kernels[ik]->getBytesAllocatedPerRep());
    checksumConsistency_width = max(checksumConsistency_width, getChecksumConsistencyName(kernels[ik]->getChecksumConsistency()).size());
    operationalComplexity_width = max(operationalComplexity_width, getComplexityName(kernels[ik]->getComplexity()).size()+3);
    maxPerfectLoopDimensions_width = max(maxPerfectLoopDimensions_width, kernels[ik]->getMaxPerfectLoopDimensions());
    problemDimensionality_width = max(problemDimensionality_width, kernels[ik]->getProblemDimensionality());
  }

  const string sepchr(" , ");

  string kernel_head("Kernel");
  kernel_width = max( kernel_head.size(),
                      kernel_width ) + 2;
  dash_width += kernel_width;

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

  double krsize = log10( static_cast<double>(kernsrep_width) );
  string kernsrep_head("Kernels/rep");
  kernsrep_width = max( static_cast<Index_type>(kernsrep_head.size()),
                        static_cast<Index_type>(krsize) ) + 3;
  dash_width += kernsrep_width + static_cast<Index_type>(sepchr.size());

  double brsize = log10( static_cast<double>(bytesMovedrep_width) );
  string bytesMovedrep_head("BytesMoved/rep");
  bytesMovedrep_width = max( static_cast<Index_type>(bytesMovedrep_head.size()),
                             static_cast<Index_type>(brsize) ) + 3;
  dash_width += bytesMovedrep_width + static_cast<Index_type>(sepchr.size());

  double frsize = log10( static_cast<double>(flopsrep_width) );
  string flopsrep_head("FLOPS/rep");
  flopsrep_width = max( static_cast<Index_type>(flopsrep_head.size()),
                         static_cast<Index_type>(frsize) ) + 3;
  dash_width += flopsrep_width + static_cast<Index_type>(sepchr.size());

  double btrsize = log10( static_cast<double>(bytesTouchedrep_width) );
  string bytesTouchedrep_head("BytesTouched/rep");
  bytesTouchedrep_width = max( static_cast<Index_type>(bytesTouchedrep_head.size()),
                        static_cast<Index_type>(btrsize) ) + 3;
  dash_width += bytesTouchedrep_width + static_cast<Index_type>(sepchr.size());

  double brrsize = log10( static_cast<double>(bytesReadrep_width) );
  string bytesReadrep_head("BytesRead/rep");
  bytesReadrep_width = max( static_cast<Index_type>(bytesReadrep_head.size()),
                        static_cast<Index_type>(brrsize) ) + 3;
  dash_width += bytesReadrep_width + static_cast<Index_type>(sepchr.size());

  double bwrsize = log10( static_cast<double>(bytesWrittenrep_width) );
  string bytesWrittenrep_head("BytesWritten/rep");
  bytesWrittenrep_width = max( static_cast<Index_type>(bytesWrittenrep_head.size()),
                        static_cast<Index_type>(bwrsize) ) + 3;
  dash_width += bytesWrittenrep_width + static_cast<Index_type>(sepchr.size());

  double bmwrsize = log10( static_cast<double>(bytesModifyWrittenrep_width) );
  string bytesModifyWrittenrep_head("BytesModifyWritten/rep");
  bytesModifyWrittenrep_width = max( static_cast<Index_type>(bytesModifyWrittenrep_head.size()),
                        static_cast<Index_type>(bmwrsize) ) + 3;
  dash_width += bytesModifyWrittenrep_width + static_cast<Index_type>(sepchr.size());

  double bamrrsize = log10( static_cast<double>(bytesAtomicModifyWrittenrep_width) );
  string bytesAtomicModifyWrittenrep_head("BytesAtomicModifyWritten/rep");
  bytesAtomicModifyWrittenrep_width = max( static_cast<Index_type>(bytesAtomicModifyWrittenrep_head.size()),
                        static_cast<Index_type>(bamrrsize) ) + 3;
  dash_width += bytesAtomicModifyWrittenrep_width + static_cast<Index_type>(sepchr.size());

  double barsize = log10( static_cast<double>(bytesAllocatedrep_width) );
  string bytesAllocatedrep_head("BytesAllocated/rep");
  bytesAllocatedrep_width = max( static_cast<Index_type>(bytesAllocatedrep_head.size()),
                                 static_cast<Index_type>(barsize) ) + 3;
  dash_width += bytesAllocatedrep_width + static_cast<Index_type>(sepchr.size());

  string checksumConsistency_head("ChecksumConsistency");
  checksumConsistency_width = max( checksumConsistency_head.size(),
                                     checksumConsistency_width ) + 2;
  dash_width += checksumConsistency_width + static_cast<Index_type>(sepchr.size());

  string operationalComplexity_head("OperationalComplexity");
  operationalComplexity_width = max( operationalComplexity_head.size(),
                                     operationalComplexity_width ) + 2;
  dash_width += operationalComplexity_width + static_cast<Index_type>(sepchr.size());

  double mpldsize = log10( static_cast<double>(maxPerfectLoopDimensions_width) );
  string maxPerfectLoopDimensions_head("MaxPerfectLoopDimensions");
  maxPerfectLoopDimensions_width = max( static_cast<Index_type>(maxPerfectLoopDimensions_head.size()),
                                        static_cast<Index_type>(mpldsize) ) + 3;
  dash_width += maxPerfectLoopDimensions_width + static_cast<Index_type>(sepchr.size());

  double pdsize = log10( static_cast<double>(problemDimensionality_width) );
  string problemDimensionality_head("ProblemDimensionality");
  problemDimensionality_width = max( static_cast<Index_type>(problemDimensionality_head.size()),
                                     static_cast<Index_type>(pdsize) ) + 3;
  dash_width += problemDimensionality_width + static_cast<Index_type>(sepchr.size());

  str           <<left << setw(kernel_width) << kernel_head
      << sepchr <<right<< setw(psize_width) << psize_head
      << sepchr <<right<< setw(reps_width) << rsize_head
      << sepchr <<right<< setw(itsrep_width) << itsrep_head
      << sepchr <<right<< setw(kernsrep_width) << kernsrep_head
      << sepchr <<right<< setw(bytesMovedrep_width) << bytesMovedrep_head
      << sepchr <<right<< setw(flopsrep_width) << flopsrep_head
      << sepchr <<right<< setw(bytesTouchedrep_width) << bytesTouchedrep_head
      << sepchr <<right<< setw(bytesReadrep_width) << bytesReadrep_head
      << sepchr <<right<< setw(bytesWrittenrep_width) << bytesWrittenrep_head
      << sepchr <<right<< setw(bytesModifyWrittenrep_width) << bytesModifyWrittenrep_head
      << sepchr <<right<< setw(bytesAtomicModifyWrittenrep_width) << bytesAtomicModifyWrittenrep_head
      << sepchr <<right<< setw(bytesAllocatedrep_width) << bytesAllocatedrep_head
      << sepchr <<left << setw(checksumConsistency_width) << checksumConsistency_head
      << sepchr <<left << setw(operationalComplexity_width) << operationalComplexity_head
      << sepchr <<left << setw(maxPerfectLoopDimensions_width) << maxPerfectLoopDimensions_head
      << sepchr <<left << setw(problemDimensionality_width) << problemDimensionality_head
      << endl;

  if ( !to_file ) {
    for (size_t i = 0; i < dash_width; ++i) {
      str << "-";
    }
    str << endl;
  }

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    KernelBase* kern = kernels[ik];
    str           <<left << setw(kernel_width) << kern->getName()
        << sepchr <<right<< setw(psize_width) << kern->getActualProblemSize()
        << sepchr <<right<< setw(reps_width) << kern->getRunReps()
        << sepchr <<right<< setw(itsrep_width) << kern->getItsPerRep()
        << sepchr <<right<< setw(kernsrep_width) << kern->getKernelsPerRep()
        << sepchr <<right<< setw(bytesMovedrep_width) << kern->getBytesMovedPerRep()
        << sepchr <<right<< setw(flopsrep_width) << kern->getFLOPsPerRep()
        << sepchr <<right<< setw(bytesTouchedrep_width) << kern->getBytesTouchedPerRep()
        << sepchr <<right<< setw(bytesReadrep_width) << kern->getBytesReadPerRep()
        << sepchr <<right<< setw(bytesWrittenrep_width) << kern->getBytesWrittenPerRep()
        << sepchr <<right<< setw(bytesModifyWrittenrep_width) << kern->getBytesModifyWrittenPerRep()
        << sepchr <<right<< setw(bytesAtomicModifyWrittenrep_width) << kern->getBytesAtomicModifyWrittenPerRep()
        << sepchr <<right<< setw(bytesAllocatedrep_width) << kern->getBytesAllocatedPerRep()
        << sepchr <<left << setw(checksumConsistency_width) << getChecksumConsistencyName(kern->getChecksumConsistency())
        << sepchr <<left << setw(operationalComplexity_width) << ("O("+getComplexityName(kern->getComplexity())+")")
        << sepchr <<right<< setw(maxPerfectLoopDimensions_width) << kern->getMaxPerfectLoopDimensions()
        << sepchr <<right<< setw(problemDimensionality_width) << kern->getProblemDimensionality()
        << endl;
  }

  str.flush();
}


void Executor::writeKernelRunDataSummary(ostream& str,
                                         vector<KernelBase*> const& kernels) const
{
  if (!str) {
    return;
  }

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  str << "Kernels run on " << num_ranks << " MPI ranks" << endl;
#else
  str << "Kernels run without MPI" << endl;
#endif

  size_t prec = 6;

//
// Set up column headers and column widths for kernel summary output.
//
  size_t     kernel_width = 0;
  size_t     variant_width = 0;
  size_t     tuning_width = 0;
  Index_type psize_width = 0;
  size_t     checksum_width = 0;
  size_t     timePerRep_width = prec + 2;
  size_t     bandwidth_width = prec + 2;
  size_t     flops_width = prec + 2;

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    kernel_width = max(kernel_width, kernels[ik]->getName().size());
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      size_t iv_width = getVariantName(variant_ids[iv]).size();
      for (std::string const& tuning_name :
           kernels[ik]->getVariantTuningNames(variant_ids[iv])) {
        variant_width = max(variant_width, iv_width);
        tuning_width = max(tuning_width, tuning_name.size());
      }
    }
    psize_width = max(psize_width, kernels[ik]->getActualProblemSize());
  }

  const string sepchr(" , ");

  string kernel_head("Kernel");
  kernel_width = max( kernel_head.size(),
                      kernel_width ) + 2;

  string variant_head("Variant");
  variant_width = max( variant_head.size(),
                      variant_width ) + 2;

  string tuning_head("Tuning");
  tuning_width = max( tuning_head.size(),
                      tuning_width ) + 2;

  double psize = log10( static_cast<double>(psize_width) );
  string psize_head("Problem size");
  psize_width = max( static_cast<Index_type>(psize_head.size()),
                     static_cast<Index_type>(psize) ) + 3;

  string checksum_head("Checksum");
  checksum_width = max( checksum_head.size(),
                        checksum_width ) + 2;

  string timePerRep_head("Mean time per rep (sec.)");
  timePerRep_width = max( timePerRep_head.size(),
                              timePerRep_width ) + 3;

  string bandwidth_head("Mean Bandwidth (GiB per sec.)");
  bandwidth_width = max( bandwidth_head.size(),
                         bandwidth_width ) + 3;

  string flops_head("Mean flops (gigaFLOP per sec.)");
  flops_width = max( flops_head.size(),
                     flops_width ) + 3;

  str           <<left << setw(kernel_width) << kernel_head
      << sepchr <<left << setw(variant_width) << variant_head
      << sepchr <<left << setw(tuning_width) << tuning_head
      << sepchr <<right<< setw(psize_width) << psize_head
      << sepchr <<left << setw(checksum_width) << checksum_head
      << sepchr <<right<< setw(timePerRep_width) << timePerRep_head
      << sepchr <<right<< setw(bandwidth_width) << bandwidth_head
      << sepchr <<right<< setw(flops_width) << flops_head
     << endl;

  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    KernelBase* kern = kernels[ik];

    Checksum_type cksum_tol = kern->getChecksumTolerance();
    Index_type problem_size = kern->getActualProblemSize();
    Index_type reps = kern->getRunReps();
    Index_type bytes_moved_per_rep = kern->getBytesMovedPerRep();
    Index_type flops_per_rep = kern->getFLOPsPerRep();

    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      VariantID vid = variant_ids[iv];
      string const& variant_name = getVariantName(vid);

      ChecksumData data = getChecksumData(kern, vid);

      for (size_t tune_idx = 0; tune_idx < kern->getNumVariantTunings(vid); ++tune_idx) {
        string const& tuning_name = kern->getVariantTuningName(vid, tune_idx);

        const char* checksum_result = "FAILED";
        if (
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
             data.checksums_rel_diff_max[tune_idx]
#else
             data.checksums_rel_diff[tune_idx]
#endif
             <= cksum_tol ) {
          checksum_result = "PASSED";
        }

        auto time_per_rep = getReportDataEntry(CSVRepMode::Timing, RunParams::CombinerOpt::Average, kern, vid, tune_idx) / reps;
        auto bandwidth = bytes_moved_per_rep / time_per_rep / (1024.0 * 1024.0 * 1024.0);
        auto flops = flops_per_rep / time_per_rep / (1000.0 * 1000.0 * 1000.0);

        str           <<left << setw(kernel_width) << kern->getName()
            << sepchr <<left << setw(variant_width) << variant_name
            << sepchr <<left << setw(tuning_width) << tuning_name
            << sepchr <<right<< setw(psize_width) << problem_size
            << sepchr <<left << setw(checksum_width) << checksum_result
            << showpoint << setprecision(prec) << std::defaultfloat
            << sepchr <<right<< setw(timePerRep_width) << time_per_rep
            << sepchr <<right<< setw(bandwidth_width) << bandwidth
            << sepchr <<right<< setw(flops_width) << flops
            << endl;
      }
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

void Executor::runKernel(KernelBase* kernel, bool print_kernel_name)
{
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  int num_ranks = -1;
  if ( run_params.showProgress() ) {
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  }
#endif

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
          const auto default_width = getCout().width();
          size_t tuning_width = std::max(size_t(12), tuning_name.size());
          getCout() << "\t\tRunning " << setw(tuning_width) << tuning_name
                    << setw(default_width) << " tuning" << flush;
        }

        kernel->execute(vid, tune_idx); // Execute kernel

        if ( run_params.showProgress() ) {
          Checksum_type cksum_tol = kernel->getChecksumTolerance();
          Checksum_type cksum_ref = kernel->getReferenceChecksum();
          Checksum_type cksum = kernel->getLastChecksum();
          Checksum_type cksum_rel_diff = KernelBase::calculateChecksumRelativeAbsoluteDifference(cksum, cksum_ref);
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
          {
            Checksum_type cksum_rel_diff_max = cksum_tol + static_cast<Checksum_type>(1e80);
            Allreduce(&cksum_rel_diff, &cksum_rel_diff_max, 1, MPI_MAX, MPI_COMM_WORLD);
            cksum_rel_diff = cksum_rel_diff_max;
          }
#endif
          const char* cksum_result = "FAILED";
          if (cksum_rel_diff <= cksum_tol) {
            cksum_result = "PASSED";
          }

          getCout() << " -- "
                    << kernel->getLastTime() / kernel->getRunReps() << " sec."
                    << " x " << kernel->getRunReps() << " rep."
                    << " " << cksum_result << " checksum"
                    << endl;
        }

      } else {
        getCout() << "\t\tSkipping " << tuning_name << " tuning" << endl;
      }

    }  // iterate over tunings 

  } // iterate over variants

}

void Executor::runWarmupKernels()
{
  RunParams::WarmupMode warmup_mode = run_params.getWarmupMode();

  if ( warmup_mode == RunParams::WarmupMode::Disable ) {
    return;
  } 

  getCout() << "\n\nRun warmup kernels...\n";

  //
  // Get warmup kernels to run from input
  //
  std::set<KernelID> warmup_kernel_ids;

  if ( warmup_mode == RunParams::WarmupMode::Explicit ) {

    warmup_kernel_ids = run_params.getSpecifiedWarmupKernelIDs();

  } else if ( warmup_mode == RunParams::WarmupMode::PerfRunSame ) {

    //
    // Warmup kernels will be same as kernels specified to run in the suite
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      warmup_kernel_ids.insert( kernel->getKernelID() );
    } // iterate over kernels to run

  } else if ( warmup_mode == RunParams::WarmupMode::Minimal ) {

    //
    // Minimal warmup mode. Choose a warmup kernel for each feature.
    //
    // First, assemble a set of feature IDs
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
    // Map feature IDs to rudimentary set of warmup kernel IDs
    //
    for ( auto fid = feature_ids.begin(); fid != feature_ids.end(); ++ fid ) {

      switch (*fid) {

        case Forall:
        case Kernel:
        case Launch:
          warmup_kernel_ids.insert(Basic_DAXPY); break;

        case Sort:
          warmup_kernel_ids.insert(Algorithm_SORT); break;

        case Scan:
          warmup_kernel_ids.insert(Basic_INDEXLIST_3LOOP); break;

        case Workgroup:
          warmup_kernel_ids.insert(Comm_HALO_PACKING_FUSED); break;

        case Reduction:
          warmup_kernel_ids.insert(Basic_REDUCE3_INT); break;

        case Atomic:
          warmup_kernel_ids.insert(Basic_PI_ATOMIC); break;

        case View:
          break;

  #ifdef RAJA_PERFSUITE_ENABLE_MPI
        case MPI:
          warmup_kernel_ids.insert(Comm_HALO_EXCHANGE_FUSED); break;
  #endif

        default:
          break;

      }

    }

  }


  //
  // Run warmup kernels
  //
  bool prev_state = KernelBase::setWarmupRun(true);

  for ( auto kid = warmup_kernel_ids.begin();
             kid != warmup_kernel_ids.end(); ++ kid ) {
    //
    // Note that we create a new kernel object for each kernel to run
    // in warmup so we don't pollute timing data, checksum data, etc.
    // for kernels that will run for real later...
    //
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

  KernelBase::setWarmupRun(prev_state);

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
  string outdir = recursiveMkdir(run_params.getOutputDirName());
  if ( !outdir.empty() ) {
#if defined(_WIN32)
    _chdir(outdir.c_str());
#else
    chdir(outdir.c_str());
#endif
  }
  string out_fprefix = "./" + run_params.getOutputFilePrefix();


  vector<FOMGroup> fom_groups;
  getFOMGroups(fom_groups);

  {
    unique_ptr<ostream> file = openOutputFile(out_fprefix + "-kernel-details.csv");
    if ( *file ) {
      constexpr bool to_file = true;
      writeKernelInfoSummary(*file, kernels, to_file);
    }

    file = openOutputFile(out_fprefix + "-kernel-run-data.csv");
    writeKernelRunDataSummary(*file, kernels);

    file = openOutputFile(out_fprefix + "-checksum.txt");
    writeChecksumReport(*file, kernels);

    for (RunParams::CombinerOpt combiner : run_params.getNpassesCombinerOpts()) {

      file = openOutputFile(out_fprefix + "-timing-" + RunParams::CombinerOptToStr(combiner) + ".csv");
      writeCSVReport(*file, kernels, CSVRepMode::Timing, combiner, 6 /* prec */);

      if ( haveReferenceVariant() ) {

        file = openOutputFile(out_fprefix + "-speedup-" + RunParams::CombinerOptToStr(combiner) + ".csv");
        writeCSVReport(*file, kernels, CSVRepMode::Speedup, combiner, 3 /* prec */);
      }
    }

    if (!fom_groups.empty() ) {
      file = openOutputFile(out_fprefix + "-fom.csv");
      writeFOMReport(*file, kernels, fom_groups);
    }
  }

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  KernelBase::setCaliperMgrFlush();
#endif

  //
  // Generate output file prefix (including directory path).
  //
  outdir = recursiveMkdir(run_params.getOutputFilePrefix()+"-per-kernel");
  if ( !outdir.empty() ) {
#if defined(_WIN32)
    _chdir(outdir.c_str());
#else
    chdir(outdir.c_str());
#endif
  }
  out_fprefix = "./" + run_params.getOutputFilePrefix();

  for (size_t i = 0; i < kernels.size(); ++i) {

    string kernel_out_fprefix = out_fprefix + "-" + kernels[i]->getName();
    vector<KernelBase*> mykernel({kernels[i]});

    unique_ptr<ostream> file = openOutputFile(kernel_out_fprefix + ".out");

    if ( *file ) {
      constexpr bool to_file = true;
      writeKernelInfoSummary(*file, mykernel, to_file);
    }

    writeSeparator(*file);
    writeKernelRunDataSummary(*file, mykernel);

    writeSeparator(*file);
    writeChecksumReport(*file, mykernel);

    for (RunParams::CombinerOpt combiner : run_params.getNpassesCombinerOpts()) {

      writeSeparator(*file);
      writeCSVReport(*file, mykernel, CSVRepMode::Timing, combiner, 6 /* prec */);

      if ( haveReferenceVariant() ) {

        writeSeparator(*file);
        writeCSVReport(*file, mykernel, CSVRepMode::Speedup, combiner, 3 /* prec */);
      }
    }

    if (!fom_groups.empty() ) {
      writeSeparator(*file);
      writeFOMReport(*file, mykernel, fom_groups);
    }

  }
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

void Executor::writeCSVReport(ostream& file,
                              vector<KernelBase*> const& kernels,
                              CSVRepMode mode,
                              RunParams::CombinerOpt combiner,
                              size_t prec)
{
  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string kernel_name_col_header_variant("Variant  ");
    const string kernel_name_col_header_tuning("Tuning  ");
    const string sepchr(" , ");

    size_t kercol_width = max(kernel_name_col_header_variant.size(),
                              kernel_name_col_header_tuning.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size());
    }
    kercol_width++;

    active_variant_tunings_type active_vartuns =
        getActiveVariantTunings(kernels, variant_ids, tuning_names);

    vector<vector<size_t>> vartuncol_width(active_vartuns.size());
    for (size_t iv = 0; iv < active_vartuns.size(); ++iv) {
      auto const& [vid, active_tuning_names] = active_vartuns[iv];
      size_t var_width = max(prec+2, getVariantName(vid).size());
      for (size_t it = 0; it < active_tuning_names.size(); ++it) {
        vartuncol_width[iv].emplace_back(max(var_width, active_tuning_names[it].size()));
      }
    }

    //
    // Print title line.
    //
    file << getReportTitle(mode, combiner);

    //
    // Wrtie CSV file contents for report.
    //

    for (size_t iv = 0; iv < active_vartuns.size(); ++iv) {
      for (size_t it = 0; it < active_vartuns[iv].second.size(); ++it) {
        file << sepchr;
      }
    }
    file << endl;

    //
    // Print column variant name line.
    //
    file <<left<< setw(kercol_width) << kernel_name_col_header_variant;
    for (size_t iv = 0; iv < active_vartuns.size(); ++iv) {
      auto const& [vid, active_tuning_names] = active_vartuns[iv];
      for (size_t it = 0; it < active_tuning_names.size(); ++it) {
        file << sepchr <<left<< setw(vartuncol_width[iv][it])
             << getVariantName(vid);
      }
    }
    file << endl;

    //
    // Print column tuning name line.
    //
    file <<left<< setw(kercol_width) << kernel_name_col_header_tuning;
    for (size_t iv = 0; iv < active_vartuns.size(); ++iv) {
      auto const& [vid, active_tuning_names] = active_vartuns[iv];
      for (size_t it = 0; it < active_tuning_names.size(); ++it) {
        file << sepchr <<left<< setw(vartuncol_width[iv][it])
             << active_tuning_names[it];
      }
    }
    file << endl;

    //
    // Print row of data for variants of each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];
      file <<left<< setw(kercol_width) << kern->getName();

      for (size_t iv = 0; iv < active_vartuns.size(); ++iv) {
        auto const& [vid, active_tuning_names] = active_vartuns[iv];
        for (size_t it = 0; it < active_tuning_names.size(); ++it) {

          file << sepchr <<right<< setw(vartuncol_width[iv][it]);
          if ( (mode == CSVRepMode::Speedup) &&
               (!kern->wasVariantTuningRun(reference_vid, reference_tune_idx) ||
                !kern->wasVariantTuningRun(vid, active_tuning_names[it])) ) {
            file << "Not run";
          } else if ( (mode == CSVRepMode::Timing) &&
                      !kern->wasVariantTuningRun(vid, active_tuning_names[it]) ) {
            file << "Not run";
          } else {
            file << setprecision(prec) << std::fixed
                 << getReportDataEntry(mode, combiner, kern, vid,
                        kern->getVariantTuningIndex(vid, active_tuning_names[it]));
          }
        }
      }
      file << endl;
    }

    file.flush();

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeFOMReport(ostream& file,
                              vector<KernelBase*> const& kernels,
                              vector<FOMGroup>& fom_groups)
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

    std::vector<active_variant_tunings_type> fom_group_vartuns(fom_groups.size());
    std::vector<size_t> fom_group_ncols(fom_groups.size(), 0);
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& fom_group = fom_groups[ifg];

      fom_group_vartuns[ifg] =
          getActiveVariantTunings(kernels, fom_group.variants, tuning_names);
      const active_variant_tunings_type& group = fom_group_vartuns[ifg];

      for (size_t gv = 0; gv < fom_group_vartuns[ifg].size(); ++gv) {
        auto const& [vid, fom_group_tuning_names] = group[gv];

        const string& variant_name = getVariantName(vid);
        // num variants and tuning
        // Includes the PM baseline and the variants and tunings to compared to it

        fom_group_ncols[ifg] += fom_group_tuning_names.size();
        for (const string& tuning_name : fom_group_tuning_names) {
          fom_col_width = max(fom_col_width, variant_name.size()+1+tuning_name.size());
        }
      }
    }

    vector< vector<int> > col_exec_count(fom_group_vartuns.size());
    vector< vector<double> > col_min(fom_group_vartuns.size());
    vector< vector<double> > col_max(fom_group_vartuns.size());
    vector< vector<double> > col_avg(fom_group_vartuns.size());
    vector< vector<double> > col_stddev(fom_group_vartuns.size());
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      col_exec_count[ifg].resize(fom_group_ncols[ifg], 0);
      col_min[ifg].resize(fom_group_ncols[ifg], numeric_limits<double>::max());
      col_max[ifg].resize(fom_group_ncols[ifg], -numeric_limits<double>::max());
      col_avg[ifg].resize(fom_group_ncols[ifg], 0.0);
      col_stddev[ifg].resize(fom_group_ncols[ifg], 0.0);
    }
    vector< vector< vector<double> > > pct_diff(kernels.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      pct_diff[ik].resize(fom_group_vartuns.size());
      for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
        pct_diff[ik][ifg].resize(fom_group_ncols[ifg], 0.0);
      }
    }

    //
    // Print title line.
    //
    file << "FOM Report : signed speedup(-)/slowdown(+) for each PM (base vs. RAJA) -> (T_RAJA - T_base) / T_base )";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t iv = 0; iv < fom_group_ncols[ifg]*2; ++iv) {
        file << sepchr;
      }
    }
    file << endl;

    file << "'OVER_TOL' in column to right if RAJA speedup is over tolerance";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t iv = 0; iv < fom_group_ncols[ifg]*2; ++iv) {
        file << sepchr;
      }
    }
    file << endl;

    string not_run("Not run");

    string pass(",        ");
    string fail(",OVER_TOL");
    string base(",base_ref");

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      const active_variant_tunings_type& group = fom_group_vartuns[ifg];
      for (size_t gv = 0; gv < group.size(); ++gv) {
        auto const& [vid, fom_group_tuning_names] = group[gv];
        string variant_name = getVariantName(vid);
        for (const string& tuning_name : fom_group_tuning_names) {
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

      for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
        const active_variant_tunings_type& group = fom_group_vartuns[ifg];

        constexpr double unknown_totTime = -1.0;
        double base_totTime = unknown_totTime;

        size_t col = 0;
        for (size_t gv = 0; gv < group.size(); ++gv) {
          auto const& [vid, fom_group_tuning_names] = group[gv];

          for (const string& tuning_name : fom_group_tuning_names) {

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

              file << sepchr <<left<< setw(fom_col_width)
                   << not_run << pass;

            }

            col++;
          }

        }  // iterate over group variants

      }  // iterate over fom_group_vartuns (i.e., columns)

      file << endl;

    } // iterate over kernels


    //
    // Compute column summary data.
    //

    // Column average...
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          col_avg[ifg][col] /= col_exec_count[ifg][col];
        }
      }
    }

    // Column standard deviation...
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];

      for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
        const active_variant_tunings_type& group = fom_group_vartuns[ifg];

        int col = 0;
        for (size_t gv = 0; gv < group.size(); ++gv) {
          auto const& [vid, fom_group_tuning_names] = group[gv];

          for (const string& tuning_name : fom_group_tuning_names) {

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

    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          col_stddev[ifg][col] /= col_exec_count[ifg][col];
        }
      }
    }

    //
    // Print column summaries.
    //
    file <<left<< setw(kercol_width) << " ";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        file << sepchr << setw(fom_col_width) <<left<< "  " <<right<< pass;
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Min";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
               << col_min[ifg][col] << pass;
        } else {
          file << sepchr <<left<< setw(fom_col_width)
               << not_run << pass;
        }
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Max";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
               << col_max[ifg][col] << pass;
        } else {
          file << sepchr <<left<< setw(fom_col_width)
               << not_run << pass;
        }
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Avg";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
               << col_avg[ifg][col] << pass;
        } else {
          file << sepchr <<left<< setw(fom_col_width)
               << not_run << pass;
        }
      }
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Std Dev";
    for (size_t ifg = 0; ifg < fom_group_vartuns.size(); ++ifg) {
      for (size_t col = 0; col < fom_group_ncols[ifg]; ++col) {
        if ( col_exec_count[ifg][col] > 0 ) {
          file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
               << col_stddev[ifg][col] << pass;
        } else {
          file << sepchr <<left<< setw(fom_col_width)
               << not_run << pass;
        }
      }
    }
    file << endl;

    file.flush();

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeSeparator(std::ostream& file)
{
  if ( file ) {
    file << endl << endl;
  }
}

void Executor::writeChecksumReport(ostream& file,
                                   std::vector<KernelBase*> const& kernels)
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
    const string dot_line("........................................................");

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

    size_t resultcol_width = 6+2;

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
    // Print column title lines.
    //
    file <<left<< setw(namecol_width) << "Kernel  " << endl;

    file << dot_line << endl;

    file <<left<< setw(namecol_width) << "Variants  "
         <<left<< setw(resultcol_width) << "Result  "
         <<left<< setw(checksum_width) << "Tolerance  "
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
         <<left<< setw(checksum_width) << "Average Checksum  "
         <<left<< setw(checksum_width) << "Max Checksum Rel Diff  "
         <<left<< setw(checksum_width) << "Checksum Rel Diff StdDev"
#else
         <<left<< setw(checksum_width) << "Checksum  "
         <<left<< setw(checksum_width) << "Checksum Rel Diff  "
#endif
         << endl;

    file <<left<< setw(namecol_width) << "  "
         <<left<< setw(resultcol_width) << "  "
         <<left<< setw(checksum_width) << "  "
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

      Checksum_type cksum_tol = kern->getChecksumTolerance();

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
        const string& variant_name = getVariantName(vid);

        ChecksumData data = getChecksumData(kern, vid);

        size_t num_tunings = kern->getNumVariantTunings(vid);
        for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
          const string& tuning_name = kern->getVariantTuningName(vid, tune_idx);

          if ( kern->wasVariantTuningRun(vid, tune_idx) ) {
            const char* result = "FAILED";
            if (
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
              data.checksums_rel_diff_max[tune_idx]
#else
              data.checksums_rel_diff[tune_idx]
#endif
               <= cksum_tol ) {
              result = "PASSED";
            }
            file <<left<< setw(namecol_width) << (variant_name+"-"+tuning_name)
                 <<left<< setw(resultcol_width) << result
                 << showpoint << setprecision(prec) << std::defaultfloat
                 <<left<< setw(checksum_width) << cksum_tol
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
                 <<left<< setw(checksum_width) << data.checksums_avg[tune_idx]
                 <<left<< setw(checksum_width) << data.checksums_rel_diff_max[tune_idx]
                 <<left<< setw(checksum_width) << data.checksums_rel_diff_stddev[tune_idx] << endl;
#else
                 <<left<< setw(checksum_width) << data.checksums[tune_idx]
                 <<left<< setw(checksum_width) << data.checksums_rel_diff[tune_idx] << endl;
#endif
          } else {
            file <<left<< setw(namecol_width) << (variant_name+"-"+tuning_name)
                 <<left<< setw(resultcol_width) << "Not Run"
                 <<left<< setw(checksum_width) << "Not Run"
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
                                         size_t tune_idx) const
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
