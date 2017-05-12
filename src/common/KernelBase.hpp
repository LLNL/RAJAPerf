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


#ifndef RAJAPerf_KernelBase_HPP
#define RAJAPerf_KernelBase_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RPTypes.hpp"
#include "common/DataUtils.hpp"

#include "RAJA/util/Timer.hpp"

#include <string>
#include <iostream>

namespace rajaperf {


class KernelBase
{
public:

  KernelBase(KernelID kid, const RunParams& params);

  virtual ~KernelBase();

  KernelID     getKernelID() const { return kernel_id; }
  const std::string& getName() const { return name; }

  Index_type getRunSize() const { return run_size; }
  Index_type getRunSamples() const { return run_samples; }

  Index_type getDefaultSize() const { return default_size; }
  Index_type getDefaultSamples() const { return default_samples; }

  SizeSpec_T getSizeSpec();

  bool wasVariantRun(VariantID vid) const 
    { return num_exec[vid] > 0; }

  double getMinTime(VariantID vid) const { return min_time[vid]; }
  double getMaxTime(VariantID vid) const { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }
  Checksum_type getChecksum(VariantID vid) const { return checksum[vid]; }

  void setDefaultSize(Index_type size); 
  void setDefaultSamples(Index_type nsamp);

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
  int num_exec[NumVariants];

  RAJA::Timer::ElapsedType min_time[NumVariants];
  RAJA::Timer::ElapsedType max_time[NumVariants];
  RAJA::Timer::ElapsedType tot_time[NumVariants];

  Checksum_type checksum[NumVariants];


private:
  KernelBase() = delete;

  void recordExecTime(); 

  KernelID    kernel_id;
  std::string name;

  RAJA::Timer timer;

  const RunParams& run_params;

  Index_type run_size;
  Index_type run_samples;

  Index_type default_size;
  Index_type default_samples;

  VariantID running_variant; 
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
