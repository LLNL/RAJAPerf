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

namespace rajaperf {

KernelBase::KernelBase(KernelID kid) 
{
  kernel_id = kid;
  name      = getKernelName(kernel_id);

  run_length      = 0;
  run_samples     = 0;
  default_length  = 0;
  default_samples = 0;

  for (size_t ivar = 0; ivar < NUM_VARIANTS; ++ivar) {
     min_time[ivar] = 0.0;
     max_time[ivar] = 0.0;
     tot_time[ivar] = 0.0;
     checksum[ivar] = 0.0;
  }
}

 
KernelBase::~KernelBase()
{
}


void KernelBase::execute(VariantID vid, const RunParams& params) 
{
  this->setUp(vid);
  
  this->executeKernel(vid, params); 

  this->computeChecksum(vid); 

  this->tearDown(vid);
}


#if 0 // RDH
void KernelBase::recordExecTime(auto start, auto end)
{
  Duration time = end - start;

  min_time = std::min(min_time, time.count());
  max_time = std::max(min_time, time.count());
  tot_time += time.count();
}
#endif

}  // closing brace for rajaperf namespace
