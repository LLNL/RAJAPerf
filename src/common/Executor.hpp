//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Executor_HPP
#define RAJAPerf_Executor_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RunParams.hpp"

#include <map>
#include <iosfwd>
#include <utility>
#include <set>

  ///////////////////////////////////////////////////
  // Logic:
  // Need the full set of kernels
  // Associate group names (e.g., lcals, basic) with kernel sets
  // Interface to add new kernels (e.g., DAXPY) and groups (basic) 
  // for Kokkos Performance Testing 
namespace rajaperf {

class KernelBase;
class WarmupKernel;

/*!
 *******************************************************************************
 *
 * \brief Class that assembles kernels and variants to run and executes them.
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

  // Interface for adding new Kokkos groups and kernels 
  using groupID = int;
  using kernelSet = std::set<KernelBase*>;
  using kernelMap = std::map<std::string, KernelBase*>;
  using groupMap =  std::map<std::string, kernelSet>;
  using kernelID = int;


  groupID registerGroup(std::string groupName);

  kernelID registerKernel(std::string, KernelBase*);

  std::vector<KernelBase*> lookUpKernelByName(std::string kernelOrGroupName);

  const RunParams& getRunParams();


private:
  Executor() = delete;

  enum CSVRepMode {
    Timing = 0,
    Speedup,

    NumRepModes // Keep this one last and DO NOT remove (!!)
  };

  struct FOMGroup {
    VariantID base;
    std::vector<VariantID> variants;
  };

  bool haveReferenceVariant() { return reference_vid < NumVariants; }

  void writeKernelInfoSummary(std::ostream& str, bool to_file) const;

  void writeCSVReport(const std::string& filename, CSVRepMode mode,
                      size_t prec);
  std::string getReportTitle(CSVRepMode mode);
  long double getReportDataEntry(CSVRepMode mode,
                                 KernelBase* kern, VariantID vid);

  void writeChecksumReport(const std::string& filename);

  void writeFOMReport(const std::string& filename);
  void getFOMGroups(std::vector<FOMGroup>& fom_groups);
  
 // Kokkos Design:
 // Kokkos add group and kernel ID inline functions
 // The newGroupID and newKerneID, both type int, will be shared amongst invocations of these inline functions.
 
  inline groupID getNewGroupID() {

        static groupID newGroupID;

        return newGroupID++;

  }

  inline kernelID getNewKernelID() {
        
        static kernelID newKernelID;
        return newKernelID++;

  }

  // Required data members:
  // running parameters, specific kernels (e.g., DAXPY), variants (e.g.,
  // Kokkos, CUDA, Sequential, etc.)

  RunParams run_params;
  std::vector<KernelBase*> kernels;
  std::vector<VariantID>   variant_ids;

  VariantID reference_vid;

  // "allKernels" is an instance of kernelMap, a std::map that takes a std::string name (key) and pointer to the associated KernelBase object (value).
  kernelMap allKernels;
  // "kernelsPerGroup" is an instance of the groupMap type, a std::map that takes a std::string name (key) and a kernelSet object,
  // containing the set of unique kernels (in a kernel group, such as basic,
  // lcals, etc.) to be run.  
  groupMap kernelsPerGroup;


};
// Kokkos design:
// Register a new kernel group (see: PerfsuiteKernelDefinitions.*):
void free_register_group(Executor*, std::string);
// Register a new kernel (that belongs to a particular kernel group):
void free_register_kernel(Executor*, std::string, KernelBase*);
// Take in run parameters by reference
const RunParams& getRunParams(Executor* exec);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
