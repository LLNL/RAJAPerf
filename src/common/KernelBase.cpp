//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "KernelBase.hpp"

#include "RunParams.hpp"

#include <cmath>

namespace rajaperf {

KernelBase::KernelBase(KernelID kid, const RunParams& params) 
  : run_params(params),
    kernel_id(kid),
    name( getFullKernelName(kernel_id) ),
    default_size(0),
    default_reps(0),
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


Index_type KernelBase::getRunSize() const
{ 
  return static_cast<Index_type>(default_size*run_params.getSizeFactor()); 
}

Index_type KernelBase::getRunReps() const
{ 
  if (run_params.getInputState() == RunParams::CheckRun) {
    return static_cast<Index_type>(run_params.getCheckRunReps());
  } else {
    return static_cast<Index_type>(default_reps*run_params.getRepFactor()); 
  } 
}


void KernelBase::execute(VariantID vid) 
{
  running_variant = vid;

  resetTimer();

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

void KernelBase::runKernel(VariantID vid)
{
  switch ( vid ) {

    case Base_Seq :
#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq :
    case RAJA_Seq :
#endif
    {
      runSeqVariant(vid);
      break;
    }

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
      runOpenMPVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  " << getName() 
                << " : Unknown variant id = " << vid << std::endl;
    }

  }
}

void KernelBase::print(std::ostream& os) const
{
  os << "\nKernelBase::print..." << std::endl;
  os << "\t\t name(id) = " << name << "(" << kernel_id << ")" << std::endl;
  os << "\t\t\t default_size = " << default_size << std::endl;
  os << "\t\t\t default_reps = " << default_reps << std::endl;
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
