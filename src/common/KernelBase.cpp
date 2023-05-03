//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "KernelBase.hpp"

#include "RunParams.hpp"
#include "OpenMPTargetDataUtils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace rajaperf {

KernelBase::KernelBase(KernelID kid, const RunParams& params)
  : run_params(params)
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  , did(getOpenMPTargetDevice())
#endif
{
  kernel_id = kid;
  name = getFullKernelName(kernel_id);

  default_prob_size = -1;
  default_reps = -1;

  actual_prob_size = -1;

  for (size_t fid = 0; fid < NumFeatures; ++fid) {
    uses_feature[fid] = false;
  }

  its_per_rep = -1;
  kernels_per_rep = -1;
  bytes_per_rep = -1;
  FLOPs_per_rep = -1;

  running_variant = NumVariants;
  running_tuning = getUnknownTuningIdx();

  checksum_scale_factor = 1.0;

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  // Init Caliper column metadata attributes 
  // Aggregatable attributes need to be initialized before manager.start()
  ProblemSize_attr = cali_create_attribute("ProblemSize", CALI_TYPE_DOUBLE,
                                           CALI_ATTR_ASVALUE | 
                                           CALI_ATTR_AGGREGATABLE | 
                                           CALI_ATTR_SKIP_EVENTS);
  Reps_attr = cali_create_attribute("Reps", CALI_TYPE_DOUBLE,
                                    CALI_ATTR_ASVALUE | 
                                    CALI_ATTR_AGGREGATABLE | 
                                    CALI_ATTR_SKIP_EVENTS);
  Iters_Rep_attr = cali_create_attribute("Iterations/Rep", CALI_TYPE_DOUBLE,
                                         CALI_ATTR_ASVALUE | 
                                         CALI_ATTR_AGGREGATABLE | 
                                         CALI_ATTR_SKIP_EVENTS);
  Kernels_Rep_attr = cali_create_attribute("Kernels/Rep", CALI_TYPE_DOUBLE,
                                           CALI_ATTR_ASVALUE | 
                                           CALI_ATTR_AGGREGATABLE | 
                                           CALI_ATTR_SKIP_EVENTS);
  Bytes_Rep_attr = cali_create_attribute("Bytes/Rep", CALI_TYPE_DOUBLE, 
                                         CALI_ATTR_ASVALUE | 
                                         CALI_ATTR_AGGREGATABLE | 
                                         CALI_ATTR_SKIP_EVENTS);
  Flops_Rep_attr = cali_create_attribute("Flops/Rep", CALI_TYPE_DOUBLE,
                                         CALI_ATTR_ASVALUE | 
                                         CALI_ATTR_AGGREGATABLE | 
                                         CALI_ATTR_SKIP_EVENTS);
#endif

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
  if (!isVariantAvailable(vid)) return;

  switch ( vid ) {

    case Base_Seq :
    {
      setSeqTuningDefinitions(vid);
      break;
    }

    case Lambda_Seq :
    case RAJA_Seq :
    {
#if defined(RUN_RAJA_SEQ)
      setSeqTuningDefinitions(vid);
#endif
      break;
    }

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      setOpenMPTuningDefinitions(vid);
#endif
      break;
    }

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
#if defined(RAJA_ENABLE_TARGET_OPENMP)
      setOpenMPTargetTuningDefinitions(vid);
#endif
      break;
    }

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
    {
#if defined(RAJA_ENABLE_CUDA)
      setCudaTuningDefinitions(vid);
#endif
      break;
    }

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
    {
#if defined(RAJA_ENABLE_HIP)
      setHipTuningDefinitions(vid);
#endif
      break;
    }
// Required for running Kokkos
    case Kokkos_Lambda :
    {
#if defined(RUN_KOKKOS)
    setKokkosTuningDefinitions(vid);
#endif
    break;
    }

    default : {
#if 0
      getCout() << "\n  " << getName()
                << " : Unknown variant id = " << vid << std::endl;
#endif
    }
  }

  checksum[vid].resize(variant_tuning_names[vid].size(), 0.0);
  num_exec[vid].resize(variant_tuning_names[vid].size(), 0);
  min_time[vid].resize(variant_tuning_names[vid].size(), std::numeric_limits<double>::max());
  max_time[vid].resize(variant_tuning_names[vid].size(), -std::numeric_limits<double>::max());
  tot_time[vid].resize(variant_tuning_names[vid].size(), 0.0);
#if defined(RAJA_PERFSUITE_USE_CALIPER)   
  doCaliMetaOnce[vid].resize(variant_tuning_names[vid].size(), true);
#endif
}

int KernelBase::getDataAlignment() const
{
  return run_params.getDataAlignment();
}

DataSpace KernelBase::getDataSpace(VariantID vid) const
{
  switch ( vid ) {

    case Base_Seq :
    case Lambda_Seq :
    case RAJA_Seq :
      return run_params.getSeqDataSpace();

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
      return run_params.getOmpDataSpace();

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
      return run_params.getOmpTargetDataSpace();

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
      return run_params.getCudaDataSpace();

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
      return run_params.getHipDataSpace();

    case Kokkos_Lambda :
      return run_params.getKokkosDataSpace();

    default:
      throw std::invalid_argument("getDataSpace : Unknown variant id");
  }
}

DataSpace KernelBase::getHostAccessibleDataSpace(VariantID vid) const
{
  return hostAccessibleDataSpace(getDataSpace(vid));
}

void KernelBase::execute(VariantID vid, size_t tune_idx)
{
  running_variant = vid;
  running_tuning = tune_idx;

  resetTimer();

  detail::resetDataInitCount();
  this->setUp(vid, tune_idx);

  this->runKernel(vid, tune_idx);

  this->updateChecksum(vid, tune_idx);

  this->tearDown(vid, tune_idx);

  running_variant = NumVariants;
  running_tuning = getUnknownTuningIdx();
}

void KernelBase::recordExecTime()
{
  num_exec[running_variant].at(running_tuning)++;

  RAJA::Timer::ElapsedType exec_time = timer.elapsed();
  min_time[running_variant].at(running_tuning) =
      std::min(min_time[running_variant].at(running_tuning), exec_time);
  max_time[running_variant].at(running_tuning) =
      std::max(max_time[running_variant].at(running_tuning), exec_time);
  tot_time[running_variant].at(running_tuning) += exec_time;
}

void KernelBase::runKernel(VariantID vid, size_t tune_idx)
{
  if ( !hasVariantDefined(vid) ) {
    return;
  }

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  if (doCaliperTiming) {
    KernelBase::setCaliperMgrStart(vid);
  }
#endif
   
  switch ( vid ) {

    case Base_Seq :
    {
      runSeqVariant(vid, tune_idx);
      break;
    }

    case Lambda_Seq :
    case RAJA_Seq :
    {
#if defined(RUN_RAJA_SEQ)
      runSeqVariant(vid, tune_idx);
#endif
      break;
    }

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
      runOpenMPVariant(vid, tune_idx);
#endif
      break;
    }

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
#if defined(RAJA_ENABLE_TARGET_OPENMP)
      runOpenMPTargetVariant(vid, tune_idx);
#endif
      break;
    }

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
    {
#if defined(RAJA_ENABLE_CUDA)
      runCudaVariant(vid, tune_idx);
#endif
      break;
    }

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
    {
#if defined(RAJA_ENABLE_HIP)
      runHipVariant(vid, tune_idx);
#endif
      break;
    }
    case Kokkos_Lambda :
    {
#if defined(RUN_KOKKOS)
      runKokkosVariant(vid, tune_idx);
#endif
    }

    default : {
#if 0
      getCout() << "\n  " << getName()
                << " : Unknown variant id = " << vid << std::endl;
#endif
    }

  }
#if defined(RAJA_PERFSUITE_USE_CALIPER)
  if (doCaliperTiming) {
    setCaliperMgrStop(vid); 
  }
#endif
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
  os << "\t\t\t variant_tuning_names: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < variant_tuning_names[j].size(); ++t) {
      os << "\t\t\t\t\t" << getVariantTuningName(static_cast<VariantID>(j), t)
                         << std::endl;
    }
  }
  os << "\t\t\t its_per_rep = " << its_per_rep << std::endl;
  os << "\t\t\t kernels_per_rep = " << kernels_per_rep << std::endl;
  os << "\t\t\t bytes_per_rep = " << bytes_per_rep << std::endl;
  os << "\t\t\t FLOPs_per_rep = " << FLOPs_per_rep << std::endl;
  os << "\t\t\t num_exec: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < num_exec[j].size(); ++t) {
      os << "\t\t\t\t\t" << num_exec[j][t] << std::endl;
    }
  }
  os << "\t\t\t min_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < min_time[j].size(); ++t) {
      os << "\t\t\t\t\t" << min_time[j][t] << std::endl;
    }
  }
  os << "\t\t\t max_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < max_time[j].size(); ++t) {
      os << "\t\t\t\t\t" << max_time[j][t] << std::endl;
    }
  }
  os << "\t\t\t tot_time: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < tot_time[j].size(); ++t) {
      os << "\t\t\t\t\t" << tot_time[j][t] << std::endl;
    }
  }
  os << "\t\t\t checksum: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < checksum[j].size(); ++t) {
      os << "\t\t\t\t\t" << checksum[j][t] << std::endl;
    }
  }
  os << std::endl;
}

#if defined(RAJA_PERFSUITE_USE_CALIPER)
void KernelBase::doOnceCaliMetaBegin(VariantID vid, size_t tune_idx)
{
  // attributes are class variables initialized in ctor
  if(doCaliMetaOnce[vid].at(tune_idx)) {
    cali_set_double(ProblemSize_attr,(double)getActualProblemSize());
    cali_set_double(Reps_attr,(double)getRunReps());
    cali_set_double(Iters_Rep_attr,(double)getItsPerRep());
    cali_set_double(Kernels_Rep_attr,(double)getKernelsPerRep());
    cali_set_double(Bytes_Rep_attr,(double)getBytesPerRep());
    cali_set_double(Flops_Rep_attr,(double)getFLOPsPerRep());
  }
}

void KernelBase::doOnceCaliMetaEnd(VariantID vid, size_t tune_idx)
{
  if(doCaliMetaOnce[vid].at(tune_idx)) {
    doCaliMetaOnce[vid].at(tune_idx) = false;
  }
}

// initialize a KernelBase static 
std::map<rajaperf::VariantID, cali::ConfigManager> KernelBase::mgr;
#endif
}  // closing brace for rajaperf namespace


