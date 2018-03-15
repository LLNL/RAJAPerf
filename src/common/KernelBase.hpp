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


#ifndef RAJAPerf_KernelBase_HPP
#define RAJAPerf_KernelBase_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RPTypes.hpp"
#include "common/DataUtils.hpp"
#include "common/RunParams.hpp"

#include "RAJA/util/Timer.hpp"

#include <string>
#include <iostream>

namespace rajaperf {

/*!
 *******************************************************************************
 *
 * \brief Pure virtual base class for all Suite kernels.
 *
 *******************************************************************************
 */
class KernelBase
{
public:

  KernelBase(KernelID kid, const RunParams& params);

  virtual ~KernelBase();

  KernelID     getKernelID() const { return kernel_id; }
  const std::string& getName() const { return name; }

  Index_type getDefaultSize() const { return default_size; }
  Index_type getDefaultReps() const { return default_reps; }

  SizeSpec_T getSizeSpec() {return run_params.getSizeSpec();}

  void setDefaultSize(Index_type size) { default_size = size; }
  void setDefaultReps(Index_type reps) { default_reps = reps; }

  Index_type getRunSize() const;
  Index_type getRunReps() const;

  bool wasVariantRun(VariantID vid) const 
    { return num_exec[vid] > 0; }

  double getMinTime(VariantID vid) const { return min_time[vid]; }
  double getMaxTime(VariantID vid) const { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }
  Checksum_type getChecksum(VariantID vid) const { return checksum[vid]; }

  void execute(VariantID vid);

  void startTimer() 
  { 
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA || running_variant == RAJA_CUDA ) {
      cudaDeviceSynchronize();
    }
#endif
    timer.start(); 
  }

  void stopTimer()  
  { 
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA || running_variant == RAJA_CUDA ) {
      cudaDeviceSynchronize();
    }
#endif
    timer.stop(); recordExecTime(); 
  }

  void resetTimer() { timer.reset(); }

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by each concrete kernel class.
  //

  virtual Index_type getItsPerRep() const { return getRunSize(); }

  virtual void print(std::ostream& os) const; 

  virtual void setUp(VariantID vid) = 0;
  virtual void runKernel(VariantID vid) = 0;
  virtual void updateChecksum(VariantID vid) = 0;
  virtual void tearDown(VariantID vid) = 0;

protected:
  int num_exec[NumVariants];

  const RunParams& run_params;

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

  Index_type default_size;
  Index_type default_reps;

  VariantID running_variant; 
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
