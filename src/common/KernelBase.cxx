/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel base class.
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


#include "KernelBase.hxx"

#include "RunParams.hxx"
#include "DataUtils.hxx"

#include <cmath>

namespace rajaperf {

KernelBase::KernelBase(KernelID kid, const RunParams& params) 
  : kernel_id(kid),
    name( getFullKernelName(kernel_id) ),
    run_params(params),
    run_size(0),
    run_samples(0),
    default_size(0),
    default_samples(0),
    running_variant(NumVariants)
{
  for (size_t ivar = 0; ivar < NumVariants; ++ivar) {
     num_exec[ivar] = 0;
     min_time[ivar] = std::numeric_limits<double>::max();
     max_time[ivar] = -std::numeric_limits<double>::max();
     tot_time[ivar] = 0.0;
     checksum[ivar] = 0.0;
  }
}

 
KernelBase::~KernelBase()
{
}


void KernelBase::setDefaultSize(int size)
{
  default_size = size;
  run_size = static_cast<int>( size*run_params.getSizeFraction() );
}

void KernelBase::setDefaultSamples(int nsamp)
{
  default_samples = nsamp;
  run_samples = static_cast<int>( nsamp*run_params.getSampleFraction() );
}

void KernelBase::execute(VariantID vid) 
{
  running_variant = vid;

  resetDataInitCount();
  this->setUp(vid);
  
  this->runKernel(vid); 

  this->updateChecksum(vid); 

  this->tearDown(vid);

  running_variant = NumVariants; 
}

void KernelBase::recordExecTime()
{
  num_exec[running_variant]++;

  RAJA::Timer::ElapsedType exec_time = timer.elapsed();
  min_time[running_variant] = std::min(min_time[running_variant], exec_time);
  max_time[running_variant] = std::max(min_time[running_variant], exec_time);
  tot_time[running_variant] += exec_time;
}

void KernelBase::print(std::ostream& os) const
{
  os << "\nKernelBase::print..." << std::endl;
  os << "\t\t name(id) = " << name << "(" << kernel_id << ")" << std::endl;
  os << "\t\t\t run_size(default_size) = " 
     << run_size << "(" << default_size << ")" << std::endl;
  os << "\t\t\t run_samples(default_samples) = " 
     << run_samples << "(" << default_samples << ")" << std::endl;
  os << "\t\t\t num_exec: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << num_exec[j] << std::endl; 
  }
  os << "\t\t\t min_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << min_time[j] << std::endl; 
  }
  os << "\t\t\t max_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << max_time[j] << std::endl; 
  }
  os << "\t\t\t tot_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << tot_time[j] << std::endl; 
  }
  os << "\t\t\t checksum: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << checksum[j] << std::endl; 
  }
  os << std::endl;
}

}  // closing brace for rajaperf namespace
