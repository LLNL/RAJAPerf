//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_RunParams_HPP
#define RAJAPerf_RunParams_HPP

#include <string>
#include <set>
#include <vector>
#include <array>
#include <iosfwd>

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"


#define ARRAY_OF_PTRS_MAX_ARRAY_SIZE 26


namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Simple class to parse and maintain suite execution parameters.
 *
 *******************************************************************************
 */
class RunParams {

public:
  RunParams( int argc, char** argv );
  ~RunParams( );

  /*!
   * \brief Enumeration indicating state of input options requested
   */
  enum InputOpt {
    InfoRequest,  /*!< option requesting information */
    DryRun,       /*!< report summary of how suite will run w/o running */
    CheckRun,     /*!< run suite with small rep count to make sure
                       everything works properly */
    PerfRun,      /*!< input defines a valid performance run,
                       suite will run as specified */
    BadInput,     /*!< erroneous input given */
    Undefined     /*!< input not defined (yet) */
  };

  /*!
   * \brief Enumeration indicating state of combiner options requested
   */
  enum CombinerOpt {
    Average,      /*!< option requesting average */
    Minimum,      /*!< option requesting minimum */
    Maximum       /*!< option requesting maximum */
  };

  static std::string CombinerOptToStr(CombinerOpt co)
  {
    switch (co) {
      case CombinerOpt::Average:
        return "Average";
      case CombinerOpt::Minimum:
        return "Minimum";
      case CombinerOpt::Maximum:
        return "Maximum";
      default:
        return "Unknown";
    }
  }

  /*!
   * \brief Enumeration indicating how to use input variables to generate
   *        problem size(s)
   */
  enum struct SizeMeaning {
    Unset,    /*!< indicates value is unset */
    Default,  /*!< Use default kernel iteration space */
    Direct,   /*!< directly use size as kernel iteration space */
    Memory    /*!< directly use memory based on memory_meaning */
  };

  /*!
   * \brief Translate SizeMeaning enum value to string
   */
  static std::string SizeMeaningToStr(SizeMeaning sm)
  {
    switch (sm) {
      case SizeMeaning::Unset:
        return "Unset";
      case SizeMeaning::Default:
        return "Default";
      case SizeMeaning::Direct:
        return "Direct";
      case SizeMeaning::Memory:
        return "Memory";
      default:
        return "Unknown";
    }
  }

  /*!
   * \brief Enumeration indicating how to interpret memory input
   */
  enum struct MemoryMeaning {
    Unset,    /*!< indicates value is unset */
    Moved,    /*!< Find problem size that matches bytesMoved/rep */
    Touched,  /*!< Find problem size that matches bytesTouched/rep */
    Allocated /*!< Find problem size that matches bytesAllocated/rep */
  };

  /*!
   * \brief Translate MemoryMeaning enum value to string
   */
  static std::string MemoryMeaningToStr(MemoryMeaning sm)
  {
    switch (sm) {
      case MemoryMeaning::Unset:
        return "Unset";
      case MemoryMeaning::Moved:
        return "Moved";
      case MemoryMeaning::Touched:
        return "Touched";
      case MemoryMeaning::Allocated:
        return "Allocated";
      default:
        return "Unknown";
    }
  }

  /*!
   * \brief Enumeration for the bin assignment algorithm used in multi-reduce kernels
   */
  enum struct BinAssignmentAlgorithm : int {
    Random,          /*!< random bin for each iterate */
    RunsRandomSizes, /*!< each bin in turn is repeated a random number of times,
                          Ex. 6 bins and 10 iterates [ 0 0 1 2 2 2 2 3 3 5] */
    RunsEvenSizes,   /*!< each bin in turn is repeated the same number of times,
                          Ex. 6 bins and 10 iterates [ 0 0 1 1 2 2 3 3 4 5] */
    Single           /*!< use bin 0 for each iterate */
  };

  /*!
   * \brief Translate BinAssignmentAlgorithm enum value to string
   */
  static std::string BinAssignmentAlgorithmToStr(BinAssignmentAlgorithm baa)
  {
    switch (baa) {
      case BinAssignmentAlgorithm::Random:
        return "Random";
      case BinAssignmentAlgorithm::RunsRandomSizes:
        return "RunsRandomSizes";
      case BinAssignmentAlgorithm::RunsEvenSizes:
        return "RunsEvenSizes";
      case BinAssignmentAlgorithm::Single:
        return "Single";
      default:
        return "Unknown";
    }
  }

  /*!
   * \brief Enumeration indicating how to run warmup kernels
   */
  enum WarmupMode {
    Disable,       /*!< no warmup kernels will be run */
    Default,       /*!< run minimal set of warmup kernels based kernels to run */
    PerfRunSame,   /*!< run warmup pass of each kernel to run */
    Explicit,      /*!< run warmup pass of each kernel explicitly named for warmup in input */
  };

  /*!
   * \brief Translate WarmupMode enum value to string
   */
  static std::string WarmupModeToStr(WarmupMode wm)
  {
    switch (wm) {
      case WarmupMode::Disable:
        return "Disable";
      case WarmupMode::Default:
        return "Default";
      case WarmupMode::PerfRunSame:
        return "PerfRunSame";
      case WarmupMode::Explicit:
        return "Explicit";
      default:
        return "Unknown";
    }
  }

  /*!
   * \brief Return state of input parsed to this point.
   */
  InputOpt getInputState() const { return input_state; }


//@{
//! @name Getters/setters for processing input and run parameters

  bool showProgress() const { return show_progress; }

  int getNumPasses() const { return npasses; }

  double getRepFactor() const { return rep_fact; }

  const std::vector<CombinerOpt>& getNpassesCombinerOpts() const
  { return npasses_combiners; }

  SizeMeaning getSizeMeaning() const { return size_meaning; }

  double getSizeFactor() const { return size_factor; }

  double getSize() const { return size; }

  MemoryMeaning getMemoryMeaning() const { return memory_meaning; }

  double getMemory() const { return memory; }

  double getMinSize() const { return min_size; }

  Index_type getTargetSize(Index_type default_prob_size) const
  {
    Index_type target_size = static_cast<Index_type>(0);
    if (size_meaning == RunParams::SizeMeaning::Default) {
      target_size = default_prob_size;
    } else if (size_meaning == RunParams::SizeMeaning::Direct) {
      target_size = static_cast<Index_type>(size);
    } else if (size_meaning == RunParams::SizeMeaning::Memory) {
      // This will be fixed up on a per kernel basis later
      target_size = static_cast<Index_type>(memory);
    }
    if (target_size < min_size) {
      target_size = static_cast<Index_type>(min_size);
    }
    target_size = static_cast<Index_type>(target_size*size_factor);
    return target_size;
  }

  Index_type getReps(Index_type default_reps) const
  {
    Index_type run_reps = static_cast<Index_type>(0);
    if (input_state == RunParams::CheckRun) {
      run_reps = static_cast<Index_type>(checkrun_reps);
    } else {
      run_reps = static_cast<Index_type>(default_reps*rep_fact);
    }
    return run_reps;
  }

  Size_type getDataAlignment() const { return data_alignment; }

  Index_type getMultiReduceNumBins() const { return multi_reduce_num_bins; }
  BinAssignmentAlgorithm getMultiReduceBinAssignmentAlgorithm() const { return multi_reduce_bin_assignment_algorithm; }

  Index_type getLtimesNumD() const { return ltimes_num_d; }
  Index_type getLtimesNumG() const { return ltimes_num_g; }
  Index_type getLtimesNumM() const { return ltimes_num_m; }

  Index_type getMCHistoryParticles() const {return MCHistoryParticles; }
  Index_type getMCHistoryCubeSz() const {return MCHistoryCubeSz; }
  Index_type getMCHistoryGroups() const {return MCHistoryGroups; }

  Index_type getFemsweepPolar() const { return femsweep_angles_polar; }
  Index_type getFemsweepAzim() const { return femsweep_angles_azim; }
  Index_type getFemsweepGroups() const { return femsweep_groups; }
  Index_type getFemsweepX() const { return femsweep_mesh_dims[0]; }
  Index_type getFemsweepY() const { return femsweep_mesh_dims[1]; }
  Index_type getFemsweepZ() const { return femsweep_mesh_dims[2]; }
  bool useFemsweepMeshDims() const { return use_femsweep_mesh_dims; }

  size_t numValidFemsweepMeshDims() const { return femsweep_mesh_dims.size(); }
  bool validFemsweepMeshDims(Index_type femsweep_mesh_dim) const
  {
    for (Index_type valid_femsweep_mesh_dims : femsweep_mesh_dims) {
      if (valid_femsweep_mesh_dims == femsweep_mesh_dim) {
        return true;
      }
    }
    return false;
  }

  Index_type getArrayOfPtrsArraySize() const { return array_of_ptrs_array_size; }

  Index_type getHaloWidth() const { return halo_width; }
  Index_type getHaloNumVars() const { return halo_num_vars; }

  bool getEnableCustomScan() const { return enable_custom_scan; }

  int getGPUStream() const { return gpu_stream; }
  size_t numValidGPUBlockSize() const { return gpu_block_sizes.size(); }
  bool validGPUBlockSize(size_t block_size) const
  {
    for (size_t valid_block_size : gpu_block_sizes) {
      if (valid_block_size == block_size) {
        return true;
      }
    }
    return false;
  }
  size_t numValidAtomicReplication() const { return atomic_replications.size(); }
  bool validAtomicReplication(size_t atomic_replication) const
  {
    for (size_t valid_atomic_replication : atomic_replications) {
      if (valid_atomic_replication == atomic_replication) {
        return true;
      }
    }
    return false;
  }
  size_t numValidItemsPerThread() const { return items_per_threads.size(); }
  bool validItemsPerThread(size_t items_per_thread) const
  {
    for (size_t valid_items_per_thread : items_per_threads) {
      if (valid_items_per_thread == items_per_thread) {
        return true;
      }
    }
    return false;
  }

  int getMPISize() const { return mpi_size; }
  int getMPIRank() const { return mpi_rank; }
  bool validMPI3DDivision() const { return (mpi_3d_division[0]*mpi_3d_division[1]*mpi_3d_division[2] == mpi_size); }
  std::array<int, 3> const& getMPI3DDivision() const { return mpi_3d_division; }

  DataSpace getSeqDataSpace() const { return seqDataSpace; }
  DataSpace getOmpDataSpace() const { return ompDataSpace; }
  DataSpace getOmpTargetDataSpace() const { return ompTargetDataSpace; }
  DataSpace getCudaDataSpace() const { return cudaDataSpace; }
  DataSpace getHipDataSpace() const { return hipDataSpace; }
  DataSpace getKokkosDataSpace() const { return kokkosDataSpace; }
  DataSpace getSyclDataSpace() const { return syclDataSpace; }

  DataSpace getSeqReductionDataSpace() const { return seqReductionDataSpace; }
  DataSpace getOmpReductionDataSpace() const { return ompReductionDataSpace; }
  DataSpace getOmpTargetReductionDataSpace() const { return ompTargetReductionDataSpace; }
  DataSpace getCudaReductionDataSpace() const { return cudaReductionDataSpace; }
  DataSpace getHipReductionDataSpace() const { return hipReductionDataSpace; }
  DataSpace getSyclReductionDataSpace() const { return syclReductionDataSpace; }
  DataSpace getKokkosReductionDataSpace() const { return kokkosReductionDataSpace; }

  DataSpace getSeqMPIDataSpace() const { return seqMPIDataSpace; }
  DataSpace getOmpMPIDataSpace() const { return ompMPIDataSpace; }
  DataSpace getOmpTargetMPIDataSpace() const { return ompTargetMPIDataSpace; }
  DataSpace getCudaMPIDataSpace() const { return cudaMPIDataSpace; }
  DataSpace getHipMPIDataSpace() const { return hipMPIDataSpace; }
  DataSpace getSyclMPIDataSpace() const { return syclMPIDataSpace; }
  DataSpace getKokkosMPIDataSpace() const { return kokkosMPIDataSpace; }

  double getPFTolerance() const { return pf_tol; }

  int getCheckRunReps() const { return checkrun_reps; }

  const std::string& getReferenceVariant() const { return reference_variant; }

  const std::vector<std::string>& getTuningInput() const
                                  { return tuning_input; }
  const std::vector<std::string>& getExcludeTuningInput() const
                                  { return exclude_tuning_input; }

  const std::string& getOutputDirName() const { return outdir; }
  const std::string& getOutputFilePrefix() const { return outfile_prefix; }

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  const std::string& getAddToSpotConfig() const { return add_to_spot_config; }
  const std::string& getAddToCaliperConfig() const { return add_to_cali_config; }
#endif

  WarmupMode getWarmupMode() const { return warmup_mode; }

  const std::set<KernelID>& getSpecifiedWarmupKernelIDs() const
    { return specified_warmup_kernel_ids; }
  const std::set<KernelID>& getKernelIDsToRun() const { return run_kernels; }
  const std::set<VariantID>& getVariantIDsToRun() const { return run_variants; }
  VariantID getReferenceVariantID() const { return reference_vid; }

//@}

  /*!
   * \brief Print all run params data to given output stream.
   */
  void print(std::ostream& str) const;


private:
  RunParams() = delete;

//@{
//! @name Routines used in command line parsing and printing option output
  void parseCommandLineOptions(int argc, char** argv);
  void printHelpMessage(std::ostream& str) const;
  void printFullKernelNames(std::ostream& str) const;
  void printKernelNames(std::ostream& str) const;
  void printVariantNames(std::ostream& str) const;
  void printDataSpaceNames(std::ostream& str) const;
  void printKernelGroupNames(std::ostream& str) const;
  void printVariantSetNames(std::ostream& str) const;
  void printFeatureNames(std::ostream& str) const;
  void printFeatureKernels(std::ostream& str) const;
  void printKernelFeatures(std::ostream& str) const;
  void printChecksumConsistencyNames(std::ostream& str) const;
  void printChecksumConsistencyKernels(std::ostream& str) const;
  void printKernelChecksumConsistencies(std::ostream& str) const;
  void printComplexityNames(std::ostream& str) const;
  void printComplexityKernels(std::ostream& str) const;
  void printKernelComplexities(std::ostream& str) const;

  void processNpassesCombinerInput();
  void processKernelInput();
  void processVariantInput();
  void processTuningInput();
//@}

  InputOpt input_state;  /*!< state of command line input */

  bool show_progress;    /*!< true -> show run progress; false -> do not */

  int npasses;           /*!< Number of passes through suite  */

  std::vector<CombinerOpt> npasses_combiners;  /*!< Combiners to use when
                              outputting timer data */

  double rep_fact;       /*!< pct of default kernel reps to run */

  SizeMeaning size_meaning; /*!< meaning of size value */
  double size_factor;    /*!< default kernel size multipier (input option) */
  double size;           /*!< kernel size to run (input option) */
  MemoryMeaning memory_meaning; /*!< meaning of memory value */
  double memory;           /*!< memory size to run (input option) */
  double min_size;           /*!< minimum kernel size to run (input option) */
  Size_type data_alignment;

  Index_type multi_reduce_num_bins; /*!< number of bins used in multi reduction kernels (input option) */
  BinAssignmentAlgorithm multi_reduce_bin_assignment_algorithm; /*!< algorithm used to assign bins to iterates used in multi reduction kernels (input option) */

  Index_type ltimes_num_d; /*!< num_d used in ltimes kernels (input option) */
  Index_type ltimes_num_g; /*!< num_g used in ltimes kernels (input option) */
  Index_type ltimes_num_m; /*!< num_m used in ltimes kernels (input option) */

  Index_type MCHistoryParticles;
  Index_type MCHistoryCubeSz;
  Index_type MCHistoryGroups;

  Index_type femsweep_angles_polar; /*!< polar angles used in femsweep kernel (input option) */
  Index_type femsweep_angles_azim; /*!< azimuthal angles used in femsweep kernel (input option) */
  Index_type femsweep_groups; /*!< groups used in femsweep kernel (input option) */
  std::vector<Index_type> femsweep_mesh_dims; /*!< mesh dimensions x, y, and z used in femsweep kernel (input option) */
  bool use_femsweep_mesh_dims; /*!< enable user input of femsweep mesh dimensions x, y, and z in femsweep kernel (true if vector femsweep_mesh_dims is properly passed on command line) */

  Index_type array_of_ptrs_array_size; /*!< number of pointers used in ARRAY_OF_PTRS kernel (input option) */

  Index_type halo_width; /*!< halo width used in halo kernels (input option) */
  Index_type halo_num_vars; /*!< num vars used in halo kernels (input option) */

  bool enable_custom_scan; /*!< enable tunings using custom scan implementations (input option) */

  int gpu_stream; /*!< 0 -> use stream 0; anything else -> use raja default stream */
  std::vector<size_t> gpu_block_sizes; /*!< Block sizes for gpu tunings to run (input option) */
  std::vector<size_t> atomic_replications; /*!< Atomic replications for gpu tunings to run (input option) */
  std::vector<size_t> items_per_threads; /*!< Items per thread for gpu tunings to run (input option) */

  int mpi_size;           /*!< Number of MPI ranks */
  int mpi_rank;           /*!< Rank of this MPI process */
  std::array<int, 3> mpi_3d_division; /*!< Number of MPI ranks in each dimension of a 3D grid */

  double pf_tol;         /*!< pct RAJA variant run time can exceed base for
                              each PM case to pass/fail acceptance */

  int checkrun_reps;     /*!< Num reps each kernel is run in check run */

  std::string reference_variant;   /*!< Name of reference variant for speedup
                                        calculations given in input */
  VariantID reference_vid;  /*!< ID of reference variant */

  DataSpace seqDataSpace = DataSpace::Host;
  DataSpace ompDataSpace = DataSpace::Omp;
  DataSpace ompTargetDataSpace = DataSpace::OmpTarget;
  DataSpace cudaDataSpace = DataSpace::CudaDevice;
  DataSpace hipDataSpace = DataSpace::HipDevice;
  DataSpace kokkosDataSpace = DataSpace::Host;
  DataSpace syclDataSpace = DataSpace::SyclDevice;

  DataSpace seqReductionDataSpace = DataSpace::Host;
  DataSpace ompReductionDataSpace = DataSpace::Omp;
  DataSpace ompTargetReductionDataSpace = DataSpace::OmpTarget;
  DataSpace cudaReductionDataSpace = DataSpace::CudaManagedDevicePreferredHostAccessed;
  DataSpace hipReductionDataSpace = DataSpace::HipDevice;
  DataSpace syclReductionDataSpace = DataSpace::SyclDevice;
  DataSpace kokkosReductionDataSpace = DataSpace::Host;

  DataSpace seqMPIDataSpace = DataSpace::Host;
  DataSpace ompMPIDataSpace = DataSpace::Omp;
  DataSpace ompTargetMPIDataSpace = DataSpace::Copy;
  DataSpace cudaMPIDataSpace = DataSpace::CudaPinned;
  DataSpace hipMPIDataSpace = DataSpace::HipPinned;
  DataSpace syclMPIDataSpace = DataSpace::SyclPinned;
  DataSpace kokkosMPIDataSpace = DataSpace::Copy;

  WarmupMode warmup_mode;

  //
  // Arrays to hold input strings for valid/invalid input. Helpful for
  // debugging command line args.
  //
  std::vector<std::string> warmup_kernel_input;
  std::vector<std::string> invalid_warmup_kernel_input;
  std::vector<std::string> kernel_input;
  std::vector<std::string> invalid_kernel_input;
  std::vector<std::string> exclude_kernel_input;
  std::vector<std::string> invalid_exclude_kernel_input;
  std::vector<std::string> variant_input;
  std::vector<std::string> invalid_variant_input;
  std::vector<std::string> exclude_variant_input;
  std::vector<std::string> invalid_exclude_variant_input;
  std::vector<std::string> tuning_input;
  std::vector<std::string> invalid_tuning_input;
  std::vector<std::string> exclude_tuning_input;
  std::vector<std::string> invalid_exclude_tuning_input;
  std::vector<std::string> feature_input;
  std::vector<std::string> invalid_feature_input;
  std::vector<std::string> exclude_feature_input;
  std::vector<std::string> invalid_exclude_feature_input;

  std::vector<std::string> npasses_combiner_input;
  std::vector<std::string> invalid_npasses_combiner_input;

  std::string outdir;          /*!< Output directory name. */
  std::string outfile_prefix;  /*!< Prefix for output data file names. */

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  std::string add_to_spot_config;
  std::string add_to_cali_config;
#endif

  std::set<KernelID>  specified_warmup_kernel_ids;
  std::set<KernelID>  run_kernels;
  std::set<VariantID> run_variants;

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
