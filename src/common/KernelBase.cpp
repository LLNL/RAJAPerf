//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "KernelBase.hpp"

#include "RunParams.hpp"

#include <cmath>

namespace rajaperf {

KernelBase::KernelBase(KernelID kid, const RunParams& params) :
  run_params(params) 
{
  kernel_id = kid;
  name = getFullKernelName(kernel_id);

  default_prob_size = -1;
  default_reps = -1;

  actual_prob_size = -1;
 
  for (size_t fid = 0; fid < NumFeatures; ++fid) {
    uses_feature[fid] = false;
  }

  for (size_t vid = 0; vid < NumVariants; ++vid) {
    has_variant_defined[vid] = false;
  }

  its_per_rep = -1;
  kernels_per_rep = -1;
  bytes_per_rep = -1;
  FLOPs_per_rep = -1;

  running_variant = NumVariants;

  checksum_scale_factor = 1.0;

  for (size_t vid = 0; vid < NumVariants; ++vid) {
    checksum[vid] = 0.0;
    num_exec[vid] = 0;
    min_time[vid] = std::numeric_limits<double>::max();
    max_time[vid] = -std::numeric_limits<double>::max();
    tot_time[vid] = 0.0;
  }
}

 
KernelBase::~KernelBase()
{
}


Index_type KernelBase::getTargetProblemSize() const
{ 
  Index_type target_size = static_cast<Index_type>(0);
  if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Factor) {
    target_size = 
      static_cast<Index_type>(default_prob_size*run_params.getSizeFactor());
  } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
    target_size = static_cast<Index_type>(run_params.getSize());
  }
  return target_size;
}

Index_type KernelBase::getRunReps() const
{ 
  Index_type run_reps = static_cast<Index_type>(0);
  if (run_params.getInputState() == RunParams::CheckRun) {
    run_reps = static_cast<Index_type>(run_params.getCheckRunReps());
  } else {
    run_reps = static_cast<Index_type>(default_reps*run_params.getRepFactor()); 
  }
  return run_reps;
}

void KernelBase::setVariantDefined(VariantID vid) 
{
  has_variant_defined[vid] = isVariantAvailable(vid); 
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
  max_time[running_variant] = std::max(max_time[running_variant], exec_time);
  tot_time[running_variant] += exec_time;
}

void KernelBase::runKernel(VariantID vid)
{
  if ( !has_variant_defined[vid] ) {
    return;
  }

  switch ( vid ) {

    case Base_Seq :
    {
      runSeqVariant(vid);
      break;
    }

    case Lambda_Seq :
    case RAJA_Seq :
    {
#if defined(RUN_RAJA_SEQ)
      runSeqVariant(vid);
#endif
      break;
    }

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      runOpenMPVariant(vid);
#endif
      break;
    }

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
#if defined(RAJA_ENABLE_TARGET_OPENMP)
      runOpenMPTargetVariant(vid);
#endif
      break;
    }

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
    {
#if defined(RAJA_ENABLE_CUDA)
      runCudaVariant(vid);
#endif
      break;
    }

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
    {
#if defined(RAJA_ENABLE_HIP)
      runHipVariant(vid);
#endif
      break;
    }

    case Base_StdPar :
    case Lambda_StdPar :
    {
      runStdParVariant(vid);
      break;
    }

    case RAJA_StdPar :
    {
#if defined(RUN_RAJA_STDPAR)
      runStdParVariant(vid);
#endif
      break;
    }


    default : {
#if 0
      std::cout << "\n  " << getName() 
                << " : Unknown variant id = " << vid << std::endl;
#endif
    }

  }
}

void KernelBase::print(std::ostream& os) const
{
  os << "\nKernelBase::print..." << std::endl;
  os << "\t\t name(id) = " << name << "(" << kernel_id << ")" << std::endl;
  os << "\t\t\t default_prob_size = " << default_prob_size << std::endl;
  os << "\t\t\t default_reps = " << default_reps << std::endl;
  os << "\t\t\t actual_prob_size = " << actual_prob_size << std::endl;
  os << "\t\t\t uses_feature: " << std::endl;
  for (unsigned j = 0; j < NumFeatures; ++j) {
    os << "\t\t\t\t" << getFeatureName(static_cast<FeatureID>(j)) 
                     << " : " << uses_feature[j] << std::endl; 
  }
  os << "\t\t\t has_variant_defined: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << has_variant_defined[j] << std::endl; 
  }
  os << "\t\t\t its_per_rep = " << its_per_rep << std::endl;
  os << "\t\t\t kernels_per_rep = " << kernels_per_rep << std::endl;
  os << "\t\t\t bytes_per_rep = " << bytes_per_rep << std::endl;
  os << "\t\t\t FLOPs_per_rep = " << FLOPs_per_rep << std::endl;
  os << "\t\t\t num_exec: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << num_exec[j] << std::endl; 
  }
  os << "\t\t\t min_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << min_time[j] << std::endl; 
  }
  os << "\t\t\t max_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << max_time[j] << std::endl; 
  }
  os << "\t\t\t tot_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << tot_time[j] << std::endl; 
  }
  os << "\t\t\t checksum: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j)) 
                     << " : " << checksum[j] << std::endl; 
  }
  os << std::endl;
}

}  // closing brace for rajaperf namespace
