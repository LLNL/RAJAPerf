//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Executor_HPP
#define RAJAPerf_Executor_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RunParams.hpp"

#include <iosfwd>
#include <utility>
#include <set>

namespace rajaperf {

class KernelBase;

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

  void writeCSVReport(const std::string& filename, CSVRepMode mode, 
                      size_t prec);
  std::string getReportTitle(CSVRepMode mode);
  long double getReportDataEntry(CSVRepMode mode, 
                                 KernelBase* kern, VariantID vid);

  void writeChecksumReport(const std::string& filename);  

  void writeFOMReport(const std::string& filename);
  void getFOMGroups(std::vector<FOMGroup>& fom_groups);
  
  RunParams run_params;
  std::vector<KernelBase*> kernels;  
  std::vector<VariantID>   variant_ids;

  VariantID reference_vid;

};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
