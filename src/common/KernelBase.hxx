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


#ifndef RAJAPerf_KernelBase_HXX
#define RAJAPerf_KernelBase_HXX

#include "common/RAJAPerfSuite.hxx"

#include "RAJA/Timer.hxx"

#include <string>
#include <iostream>

namespace rajaperf {

//
// Volatile index type for kernel sampling loops. If this is not used, 
// some kernels may be optimized away.
//
typedef volatile int SampIndex_type;


class KernelBase
{
public:

  KernelBase(KernelID kid, const RunParams& params);

  virtual ~KernelBase();

  KernelID     getKernelID() const { return kernel_id; }
  const std::string& getName() const { return name; }

  int getRunSize() const { return run_size; }
  int getRunSamples() const { return run_samples; }

  int getDefaultSize() const { return default_size; }
  int getDefaultSamples() const { return default_samples; }

  double getMinTime(VariantID vid) { return min_time[vid]; }
  double getMaxTime(VariantID vid) { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }
  long double getChecksum(VariantID vid) { return checksum[vid]; }

  void setDefaultSize(int size); 
  void setDefaultSamples(int nsamp);

  void execute(VariantID vid);
  void startTimer() { timer.start(); }
  void stopTimer()  { timer.stop(); recordExecTime(); }
  void resetTimer() { timer.reset(); }

  virtual void print(std::ostream& os) const; 

  virtual void setUp(VariantID vid) = 0;
  virtual void runKernel(VariantID vid) = 0;
  virtual void updateChecksum(VariantID vid) = 0;
  virtual void tearDown(VariantID vid) = 0;

protected:
  RAJA::Timer::ElapsedType min_time[NumVariants];
  RAJA::Timer::ElapsedType max_time[NumVariants];
  RAJA::Timer::ElapsedType tot_time[NumVariants];

  long double checksum[NumVariants];

private:
  KernelBase() = delete;

  void recordExecTime(); 

  KernelID    kernel_id;
  std::string name;

  RAJA::Timer timer;

  const RunParams& run_params;

  int run_size;
  int run_samples;

  int default_size;
  int default_samples;

  VariantID running_variant; 
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
