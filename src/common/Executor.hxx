/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing executor class that runs suite.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Executor_HXX
#define RAJAPerf_Executor_HXX

#include "common/RAJAPerfSuite.hxx"
#include "common/RunParams.hxx"

#include <iosfwd>

namespace rajaperf {

class KernelBase;

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

  void processRunData();
  void writeTimingReport(const std::string& out_fname);
  void writeChecksumReport(const std::string& out_fname);

  RunParams run_params;
  std::vector<KernelBase*> kernels;  
  std::vector<VariantID>   variants;

  VariantID baseline_variant;

};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
