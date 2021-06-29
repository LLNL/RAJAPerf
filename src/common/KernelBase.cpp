//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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

  default_size = -1;
  default_reps = -1;
 
  for (size_t fid = 0; fid < NumFeatures; ++fid) {
    uses_feature[fid] = false;
  }

  for (size_t vid = 0; vid < NumVariants; ++vid) {
    has_variant_defined[vid] = false;
  }

  problem_size = -1;
  its_per_rep = -1;
  kernels_per_rep = -1;
  FLOPs_per_rep = -1;

  running_variant = NumVariants;

  for (size_t vid = 0; vid < NumVariants; ++vid) {
    checksum[vid] = 0.0;
    num_exec[vid] = 0;
    min_time[vid] = std::numeric_limits<double>::max();
    max_time[vid] = -std::numeric_limits<double>::max();
    tot_time[vid] = 0.0;
  }
}

    KernelBase::KernelBase(std::string name, const RunParams& params)
            : run_params(params),
              kernel_id(Basic_DAXPY), // TODO DZP: better
              name(name),
              default_size(0),
              default_reps(0),
              running_variant(NumVariants)
    {
        for (size_t ivar = 0; ivar < NumVariants; ++ivar) {
            checksum[ivar] = 0.0;
            num_exec[ivar] = 0;
            min_time[ivar] = std::numeric_limits<double>::max();
            max_time[ivar] = -std::numeric_limits<double>::max();
            tot_time[ivar] = 0.0;
            has_variant_to_run[ivar] = false;
        }
    }

 
KernelBase::~KernelBase()
{
}


Index_type KernelBase::getRunSize() const
{ 
  Index_type run_size = static_cast<Index_type>(0);
  if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Factor) {
    run_size = static_cast<Index_type>(default_size*run_params.getSizeFactor());
  } else if (run_params.getSizeMeaning() == RunParams::SizeMeaning::Direct) {
    run_size = static_cast<Index_type>(run_params.getSize());
  }
  return run_size;
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
#ifdef RUN_KOKKOS 
  Kokkos::Tools::pushRegion(this->getName() + ":"+getVariantName(vid));
#endif
  this->runKernel(vid); 
#ifdef RUN_KOKKOS 
  Kokkos::Tools::popRegion();
#endif

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

#if defined(RUN_KOKKOS)
    case Kokkos_Lambda :
    case Kokkos_Functor :
    {
      runKokkosVariant(vid);
      break;
    }
#endif // RUN_KOKKOS
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
