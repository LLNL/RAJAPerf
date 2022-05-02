//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Executor_HPP
#define RAJAPerf_Executor_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RunParams.hpp"

#include <iosfwd>
#include <streambuf>
#include <memory>
#include <utility>
#include <set>

namespace rajaperf {

class KernelBase;
class WarmupKernel;

/*!
 *******************************************************************************
 *
 * \brief Class that assembles kernels, variants, and tunings to run and
 *        executes them.
 *
 *******************************************************************************
 */
class Executor
{
public:
  Executor( int argc, char** argv );

  ~Executor();

  void setupSuite();

  void reportRunSummary(std::ostream& str) const;

  void runSuite();

  void outputRunData();

private:
  Executor() = delete;

  enum CSVRepMode {
    Timing = 0,
    Speedup,

    NumRepModes // Keep this one last and DO NOT remove (!!)
  };

  struct FOMGroup {
    std::vector<VariantID> variants;
  };

  template < typename Kernel >
  KernelBase* makeKernel();

  void runKernel(KernelBase* kern, bool print_kernel_name);

  std::unique_ptr<std::ostream> openOutputFile(const std::string& filename) const;

  bool haveReferenceVariant() { return reference_vid < NumVariants; }

  void writeKernelInfoSummary(std::ostream& str, bool to_file) const;

  void writeCSVReport(std::ostream& file, CSVRepMode mode,
                      RunParams::CombinerOpt combiner, size_t prec);
  std::string getReportTitle(CSVRepMode mode, RunParams::CombinerOpt combiner);
  long double getReportDataEntry(CSVRepMode mode, RunParams::CombinerOpt combiner,
                                 KernelBase* kern, VariantID vid, size_t tune_idx);

  void writeChecksumReport(std::ostream& file);

  void writeFOMReport(std::ostream& file, std::vector<FOMGroup>& fom_groups);
  void getFOMGroups(std::vector<FOMGroup>& fom_groups);

  RunParams run_params;
  std::vector<KernelBase*> kernels;
  std::vector<VariantID>   variant_ids;
  std::vector<std::string> tuning_names[NumVariants];

  VariantID reference_vid;
  size_t    reference_tune_idx;

public:
  // Methods for verification testing in CI.
  std::vector<KernelBase*> getKernels() const { return kernels; }
  std::vector<VariantID> getVariantIDs() const { return variant_ids; }

};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
