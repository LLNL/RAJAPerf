//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_RunParams_HPP
#define RAJAPerf_RunParams_HPP

#include <string>
#include <set>
#include <vector>
#include <iosfwd>

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"

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

  int getNumPasses() const { return npasses; }

  double getRepFactor() const { return rep_fact; }

  const std::vector<CombinerOpt>& getNpassesCombinerOpts() const
  { return npasses_combiners; }

  SizeMeaning getSizeMeaning() const { return size_meaning; }

  double getSize() const { return size; }

  double getSizeFactor() const { return size_factor; }

  Size_type getDataAlignment() const { return data_alignment; }

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

  DataSpace getSeqDataSpace() const { return seqDataSpace; }
  DataSpace getOmpDataSpace() const { return ompDataSpace; }
  DataSpace getOmpTargetDataSpace() const { return ompTargetDataSpace; }
  DataSpace getCudaDataSpace() const { return cudaDataSpace; }
  DataSpace getHipDataSpace() const { return hipDataSpace; }
  DataSpace getKokkosDataSpace() const { return kokkosDataSpace; }

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
#endif

  bool getDisableWarmup() const { return disable_warmup; }

  bool getAllowProblematicImplementations() const
  { return allow_problematic_implementations; }

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
  void printGroupNames(std::ostream& str) const;
  void printFeatureNames(std::ostream& str) const;
  void printFeatureKernels(std::ostream& str) const;
  void printKernelFeatures(std::ostream& str) const;

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
  double size;           /*!< kernel size to run (input option) */
  double size_factor;    /*!< default kernel size multipier (input option) */
  Size_type data_alignment;

  int gpu_stream; /*!< 0 -> use stream 0; anything else -> use raja default stream */
  std::vector<size_t> gpu_block_sizes; /*!< Block sizes for gpu tunings to run (input option) */

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

  //
  // Arrays to hold input strings for valid/invalid input. Helpful for
  // debugging command line args.
  //
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
#endif

  bool disable_warmup;

  bool allow_problematic_implementations;

  std::set<KernelID>  run_kernels;
  std::set<VariantID> run_variants;

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
