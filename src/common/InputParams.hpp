//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_InputParams_HPP
#define RAJAPerf_InputParams_HPP

#include <string>
#include <set>
#include <vector>
#include <array>
#include <iosfwd>

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"

namespace rajaperf
{

class CommonParams {

public:
  int checkrun_reps{0};   /*!< Num reps each kernel is run in check run */ // 1
  double rep_fact{1.0};   /*!< pct of default kernel reps to run */ // 1.0

  double size{0.0};                             /*!< kernel size to run (input option) */
  double size_factor{0.0};                      /*!< default kernel size multipier (input option) */
  Size_type data_alignment{RAJA::DATA_ALIGN};

  int gpu_stream{1};                       /*!< 0 -> use stream 0; anything else -> use raja default stream */
  std::vector<size_t> gpu_block_sizes;     /*!< Block sizes for gpu tunings to run (input option) */
  std::vector<size_t> atomic_replications; /*!< Atomic replications for gpu tunings to run (input option) */
  std::vector<size_t> items_per_threads;   /*!< Items per thread for gpu tunings to run (input option) */

  int mpi_size{1};                                  /*!< Number of MPI ranks */
  int mpi_rank{0};                                  /*!< Rank of this MPI process */
  std::array<int, 3> mpi_3d_division{{-1, -1, -1}}; /*!< Number of MPI ranks in each dimension of a 3D grid */

  DataSpace seqDataSpace = DataSpace::Host;
  DataSpace ompDataSpace = DataSpace::Omp;
  DataSpace ompTargetDataSpace = DataSpace::OmpTarget;
  DataSpace cudaDataSpace = DataSpace::CudaDevice;
  DataSpace hipDataSpace = DataSpace::HipDevice;
  DataSpace kokkosDataSpace = DataSpace::Host;

  DataSpace seqReductionDataSpace = DataSpace::Host;
  DataSpace ompReductionDataSpace = DataSpace::Omp;
  DataSpace ompTargetReductionDataSpace = DataSpace::OmpTarget;
  DataSpace cudaReductionDataSpace = DataSpace::CudaManagedDevicePreferredHostAccessed;
  DataSpace hipReductionDataSpace = DataSpace::HipDevice;
  DataSpace kokkosReductionDataSpace = DataSpace::Host;

  DataSpace seqMPIDataSpace = DataSpace::Host;
  DataSpace ompMPIDataSpace = DataSpace::Omp;
  DataSpace ompTargetMPIDataSpace = DataSpace::Copy;
  DataSpace cudaMPIDataSpace = DataSpace::CudaPinned;
  DataSpace hipMPIDataSpace = DataSpace::HipPinned;
  DataSpace kokkosMPIDataSpace = DataSpace::Copy;
};

/*!
 *******************************************************************************
 *
 * \brief Simple class to parse and maintain suite execution parameters.
 *
 *******************************************************************************
 */
class InputParams : CommonParams {

public:
  InputParams( int argc, char** argv );
  ~InputParams();

  InputParams() = delete;
  InputParams(InputParams const&) = delete;
  InputParams& operator=(InputParams const&) = delete;
  InputParams(InputParams &&) = delete;
  InputParams& operator=(InputParams &&) = delete;

  /*!
   * \brief Enumeration indicating state of input options requested
   */
  enum InputOpt {
    InfoRequest,  /*!< option requesting information */
    DryRun,       /*!< report summary of how suite will run w/o running */
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
   * \brief Enumeration indicating how to interpret size input
   */
  enum SizeMeaning {
    Unset,    /*!< indicates value is unset */
    Factor,   /*!< multiplier on default kernel iteration space */
    Direct,   /*!< directly use as kernel iteration space */
  };

  /*!
   * \brief Translate SizeMeaning enum value to string
   */
  static std::string SizeMeaningToStr(SizeMeaning sm)
  {
    switch (sm) {
      case SizeMeaning::Unset:
        return "Unset";
      case SizeMeaning::Factor:
        return "Factor";
      case SizeMeaning::Direct:
        return "Direct";
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

  bool getDisableWarmup() const { return disable_warmup; }

  int getNumPasses() const { return npasses; }
  const std::vector<CombinerOpt>& getNpassesCombinerOpts() const
  { return npasses_combiners; }

  double getPFTolerance() const { return pf_tol; }
  const std::string& getReferenceVariant() const { return reference_variant; }
  VariantID getReferenceVariantID() const { return reference_vid; }

  SizeMeaning getSizeMeaning() const { return size_meaning; }

  CommonParams const& getCommonParams() const { return *this; }

  const std::string& getOutputDirName() const { return outdir; }
  const std::string& getOutputFilePrefix() const { return outfile_prefix; }

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  const std::string& getAddToSpotConfig() const { return add_to_spot_config; }
#endif

  const std::vector<std::string>& getTuningInput() const
                                  { return tuning_input; }
  const std::vector<std::string>& getExcludeTuningInput() const
                                  { return exclude_tuning_input; }


  const std::set<KernelID>& getKernelIDsToRun() const { return run_kernels; }
  const std::set<VariantID>& getVariantIDsToRun() const { return run_variants; }

//@}

  /*!
   * \brief Print all input params data to given output stream.
   */
  void print(std::ostream& str) const;


private:

//@{
//! @name Routines used in command line parsing and printing option output
  void parseCommandLineOptions(int argc, char** argv);
  void printHelpMessage(std::ostream& str) const;
  void printFullKernelNames(std::ostream& str) const;
  void printKernelNames(std::ostream& str) const;
  void printVariantNames(std::ostream& str) const;
  void printDataSpaceNames(std::ostream& str) const;
  void printGroupNames(std::ostream& str) const;
  void printFeatureNames(std::ostream& str) const;
  void printFeatureKernels(std::ostream& str) const;
  void printKernelFeatures(std::ostream& str) const;

  void processNpassesCombinerInput();
  void processKernelInput();
  void processVariantInput();
  void processTuningInput();
//@}

  InputOpt input_state{InputOpt::Undefined};   /*!< state of command line input */

  bool show_progress{false};   /*!< true -> show run progress; false -> do not */

  bool disable_warmup{false};

  int npasses{1};                              /*!< Number of passes through suite  */
  std::vector<CombinerOpt> npasses_combiners;  /*!< Combiners to use when
                                                    outputting timer data */

  double pf_tol{0.1};                    /*!< pct RAJA variant run time can exceed base for
                                              each PM case to pass/fail acceptance */
  std::string reference_variant;         /*!< Name of reference variant for speedup
                                              calculations given in input */
  VariantID reference_vid{NumVariants};  /*!< ID of reference variant */

  SizeMeaning size_meaning{SizeMeaning::Unset}; /*!< meaning of size value */

  std::string outdir;                      /*!< Output directory name. */
  std::string outfile_prefix{"RAJAPerf"};  /*!< Prefix for output data file names. */

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  std::string add_to_spot_config;
#endif

  //
  // Arrays to hold input strings for valid/invalid input. Helpful for
  // debugging command line args.
  //
  std::vector<std::string> npasses_combiner_input;
  std::vector<std::string> invalid_npasses_combiner_input;
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

  std::set<KernelID>  run_kernels;
  std::set<VariantID> run_variants;

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
