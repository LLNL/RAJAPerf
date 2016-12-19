/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing base class that defines API and 
 *          common implementation for each kernel in suite.
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


#ifndef RAJAPerfKernelBase_HXX

#include "common/RAJAPerfSuite.hxx"

#include <chrono>
#include <string>

namespace rajaperf {

class RunParams;

class KernelBase
{
public:

#if 0 // RDH
  using clock = std::chrono::steady_clock;
  using TimeType = clock::time_point;
  using Duration = std::chrono::duration<double>;
#endif

  explicit KernelBase(KernelID kid);

  virtual ~KernelBase();

  KernelID     getKernelID() { return kernel_id; }
  std::string& getName() { return name; }

  double getMinTime(VariantID vid) { return min_time[vid]; }
  double getMaxTime(VariantID vid) { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }

  void execute(VariantID vid, const RunParams& params);
#if 0 // RDH
  void recordExecTime(auto start, auto end); 
#endif

  virtual void setUp(VariantID vid) = 0;
  virtual void executeKernel(VariantID vid, const RunParams& params) = 0;
  virtual void computeChecksum(VariantID vid) = 0;
  virtual void tearDown(VariantID vid) = 0;

protected:
  KernelID    kernel_id;
  std::string name;

  int run_length;
  int run_samples;

  int default_length;
  int default_samples;

  double min_time[NUM_VARIANTS];
  double max_time[NUM_VARIANTS];
  double tot_time[NUM_VARIANTS];

  long double checksum[NUM_VARIANTS];

private:
   KernelBase() = delete;
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
