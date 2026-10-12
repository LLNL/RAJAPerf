//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "KernelBase.hpp"

#include "RunParams.hpp"
#include "OpenMPTargetDataUtils.hpp"

#include "RAJA/RAJA.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <regex>

namespace rajaperf {

//
// Static method to set whether kernels are used for warmup purposes or not
//
bool KernelBase::setWarmupRun(bool warmup_run)
{
  bool previous_state = s_warmup_run;
  s_warmup_run = warmup_run;
  return previous_state;
}

KernelBase::KernelBase(KernelID kid, const RunParams& params)
  : run_params(params)
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  , did(getOpenMPTargetDevice())
  , hid(getOpenMPTargetHost())
#endif
{
  kernel_id = kid;
  name = getFullKernelName(kernel_id);

  checksum_tolerance = ChecksumTolerance::normal;

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  // Init Caliper column metadata attributes
  // Aggregatable attributes need to be initialized before manager.start()
  ProblemSize_attr = cali_create_attribute("ProblemSize", CALI_TYPE_INT,
                                           CALI_ATTR_ASVALUE |
                                           CALI_ATTR_AGGREGATABLE |
                                           CALI_ATTR_SKIP_EVENTS);
  Reps_attr = cali_create_attribute("Reps", CALI_TYPE_INT,
                                    CALI_ATTR_ASVALUE |
                                    CALI_ATTR_AGGREGATABLE |
                                    CALI_ATTR_SKIP_EVENTS);
  Iters_Rep_attr = cali_create_attribute("Iterations/Rep", CALI_TYPE_INT,
                                         CALI_ATTR_ASVALUE |
                                         CALI_ATTR_AGGREGATABLE |
                                         CALI_ATTR_SKIP_EVENTS);
  Kernels_Rep_attr = cali_create_attribute("Kernels/Rep", CALI_TYPE_INT,
                                           CALI_ATTR_ASVALUE |
                                           CALI_ATTR_AGGREGATABLE |
                                           CALI_ATTR_SKIP_EVENTS);
  Bytes_Allocated_Rep_attr = cali_create_attribute("BytesAllocated/Rep", CALI_TYPE_INT,
                                         CALI_ATTR_ASVALUE |
                                         CALI_ATTR_AGGREGATABLE |
                                         CALI_ATTR_SKIP_EVENTS);
  Bytes_Moved_Rep_attr = cali_create_attribute("BytesMoved/Rep", CALI_TYPE_INT,
                                         CALI_ATTR_ASVALUE |
                                         CALI_ATTR_AGGREGATABLE |
                                         CALI_ATTR_SKIP_EVENTS);
  Bytes_Touched_Rep_attr = cali_create_attribute("BytesTouched/Rep", CALI_TYPE_INT,
                                                 CALI_ATTR_ASVALUE |
                                                 CALI_ATTR_AGGREGATABLE |
                                                 CALI_ATTR_SKIP_EVENTS);
  Bytes_Read_Rep_attr = cali_create_attribute("BytesRead/Rep", CALI_TYPE_INT,
                                              CALI_ATTR_ASVALUE |
                                              CALI_ATTR_AGGREGATABLE |
                                              CALI_ATTR_SKIP_EVENTS);
  Bytes_Written_Rep_attr = cali_create_attribute("BytesWritten/Rep", CALI_TYPE_INT,
                                                 CALI_ATTR_ASVALUE |
                                                 CALI_ATTR_AGGREGATABLE |
                                                 CALI_ATTR_SKIP_EVENTS);
  Bytes_ModifyWritten_Rep_attr = cali_create_attribute("BytesModifyWritten/Rep", CALI_TYPE_INT,
                                                       CALI_ATTR_ASVALUE |
                                                       CALI_ATTR_AGGREGATABLE |
                                                       CALI_ATTR_SKIP_EVENTS);
  Bytes_AtomicModifyWritten_Rep_attr = cali_create_attribute("BytesAtomicModifyWritten/Rep", CALI_TYPE_INT,
                                                             CALI_ATTR_ASVALUE |
                                                             CALI_ATTR_AGGREGATABLE |
                                                             CALI_ATTR_SKIP_EVENTS);
  Flops_Rep_attr = cali_create_attribute("Flops/Rep", CALI_TYPE_INT,
                                         CALI_ATTR_ASVALUE |
                                         CALI_ATTR_AGGREGATABLE |
                                         CALI_ATTR_SKIP_EVENTS);
  BlockSize_attr = cali_create_attribute("BlockSize", CALI_TYPE_INT,
                                           CALI_ATTR_ASVALUE |
                                           CALI_ATTR_AGGREGATABLE |
                                           CALI_ATTR_SKIP_EVENTS);
  for (unsigned i = 0; i < FeatureID::NumFeatures; ++i) {
    FeatureID fid = static_cast<FeatureID>(i);
    std::string feature = getFeatureName(fid);
    Feature_attrs[feature] = cali_create_attribute(feature.c_str(), CALI_TYPE_INT,
                                              CALI_ATTR_ASVALUE |
                                              CALI_ATTR_AGGREGATABLE |
                                              CALI_ATTR_SKIP_EVENTS);
  }
  ChecksumConsistency_attr = cali_create_attribute("ChecksumConsistency", CALI_TYPE_STRING,
                                                   CALI_ATTR_SKIP_EVENTS);
  Complexity_attr = cali_create_attribute("Complexity", CALI_TYPE_STRING,
                                           CALI_ATTR_SKIP_EVENTS);
  MaxPerfectLoopDimensions_attr = cali_create_attribute("MaxPerfectLoopDimensions", CALI_TYPE_INT,
                                           CALI_ATTR_ASVALUE |
                                           CALI_ATTR_AGGREGATABLE |
                                           CALI_ATTR_SKIP_EVENTS);
  ProblemDimensionality_attr = cali_create_attribute("ProblemDimensionality", CALI_TYPE_INT,
                                           CALI_ATTR_ASVALUE |
                                           CALI_ATTR_AGGREGATABLE |
                                           CALI_ATTR_SKIP_EVENTS);
#if defined(RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL)
  Subkernel_attr = cali_create_attribute("subkernel", CALI_TYPE_STRING,
                                         CALI_ATTR_NESTED);
#endif
#endif
}


KernelBase::~KernelBase()
{
}


void KernelBase::addVariantTuning(VariantID vid, std::string name,
                                  TuningAttribute attrs,
                                  variant_tuning_method_pointer method)
{
  if (!isVariantAvailable(vid)) return;

  variant_tuning_names[vid].emplace_back(std::move(name));
  variant_tuning_attrs[vid].emplace_back(std::move(attrs));
  variant_tuning_methods[vid].emplace_back(method);
  checksum_min[vid].emplace_back(std::numeric_limits<Checksum_type>::max());
  checksum_max[vid].emplace_back(-std::numeric_limits<Checksum_type>::max());
  checksum_sum[vid].emplace_back(0.0);
  num_exec[vid].emplace_back(0);
  min_time[vid].emplace_back(std::numeric_limits<double>::max());
  max_time[vid].emplace_back(-std::numeric_limits<double>::max());
  tot_time[vid].emplace_back(0.0);
#if defined(RAJA_PERFSUITE_USE_CALIPER)
  doCaliMetaOnce[vid].emplace_back(true);
#endif
}

Size_type KernelBase::getDataAlignment() const
{
  return run_params.getDataAlignment();
}

Size_type KernelBase::getNBytesPaddedToDataAlignment(Size_type size) const
{
  Size_type misalignment = size % run_params.getDataAlignment();
  if (misalignment) {
    size += run_params.getDataAlignment() - misalignment;
  }
  return size;
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

    case Base_SYCL :
    case RAJA_SYCL :
      return run_params.getSyclDataSpace();

    case Kokkos_Lambda :
      return run_params.getKokkosDataSpace();

    default:
      throw std::invalid_argument("getDataSpace : Unknown variant id");
  }
}

DataSpace KernelBase::getMPIDataSpace(VariantID vid) const
{
  switch ( vid ) {

    case Base_Seq :
    case Lambda_Seq :
    case RAJA_Seq :
      return run_params.getSeqMPIDataSpace();

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
      return run_params.getOmpMPIDataSpace();

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
      return run_params.getOmpTargetMPIDataSpace();

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
      return run_params.getCudaMPIDataSpace();

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
      return run_params.getHipMPIDataSpace();

    case Base_SYCL :
    case RAJA_SYCL :
      return run_params.getSyclMPIDataSpace();

    case Kokkos_Lambda :
      return run_params.getKokkosMPIDataSpace();

    default:
      throw std::invalid_argument("getDataSpace : Unknown variant id");
  }
}

DataSpace KernelBase::getReductionDataSpace(VariantID vid) const
{
  switch ( vid ) {

    case Base_Seq :
    case Lambda_Seq :
    case RAJA_Seq :
      return run_params.getSeqReductionDataSpace();

    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
      return run_params.getOmpReductionDataSpace();

    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
      return run_params.getOmpTargetReductionDataSpace();

    case Base_CUDA :
    case Lambda_CUDA :
    case RAJA_CUDA :
      return run_params.getCudaReductionDataSpace();

    case Base_HIP :
    case Lambda_HIP :
    case RAJA_HIP :
      return run_params.getHipReductionDataSpace();

    case Base_SYCL :
    case RAJA_SYCL :
      return run_params.getSyclReductionDataSpace();

    case Kokkos_Lambda :
      return run_params.getKokkosReductionDataSpace();

    default:
      throw std::invalid_argument("getReductionDataSpace : Unknown variant id");
  }
}

void KernelBase::execute(VariantID vid, size_t tune_idx)
{
  running_variant = vid;
  running_tuning = tune_idx;
  TuningAttribute running_attrs = getTuningAttributes(vid, tune_idx);

  resetTimer();

  detail::resetDataInitCount();
  this->setUp(vid, tune_idx);

  this->runKernel(vid, tune_idx);

  checksum.reset();
  this->updateChecksum(vid, tune_idx);
  Checksum_type new_checksum = getLastChecksum();

  checksum_min[vid].at(tune_idx) = std::min(new_checksum, checksum_min[vid].at(tune_idx));
  checksum_max[vid].at(tune_idx) = std::max(new_checksum, checksum_max[vid].at(tune_idx));
  checksum_sum[vid].at(tune_idx) += new_checksum;

  if ( checksum_reference_variant == NumVariants ||
       ( !hasTuningAttribute(checksum_reference_tuning_attributes, TuningAttribute::preferred_checksum) &&
         hasTuningAttribute(running_attrs, TuningAttribute::preferred_checksum) ) ) {
    // use first run variant tuning as checksum reference
    // or the first run variant tuning with preferred_checksum
    checksum_reference = new_checksum;
    checksum_reference_variant = vid;
    checksum_reference_tuning = tune_idx;
    checksum_reference_tuning_attributes = running_attrs;
  }

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
    KernelBase::setCaliperMgrStart(vid, getVariantTuningName(vid, tune_idx));
  }
#endif

  (this->*(variant_tuning_methods[vid].at(tune_idx)))(vid);

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  if (doCaliperTiming) {
    KernelBase::setCaliperMgrStop(vid, getVariantTuningName(vid, tune_idx));
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
  os << "\t\t\t checksum_consistency = " << getChecksumConsistencyName(checksum_consistency) << std::endl;
  os << "\t\t\t algorithmic_complexity = " << getComplexityName(complexity) << std::endl;
  os << "\t\t\t number_max_nested_perfect_loop_levels = " << num_nested_perfect_loops << std::endl;
  os << "\t\t\t problem_dimensionality = " << problem_dimensionality << std::endl;
  os << "\t\t\t variant_tuning_names: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < variant_tuning_names[j].size(); ++t) {
      os << "\t\t\t\t\t" << getVariantTuningName(static_cast<VariantID>(j), t)
                         << std::endl;
    }
  }
  os << "\t\t\t variant_tuning_attrs: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < variant_tuning_attrs[j].size(); ++t) {
      os << "\t\t\t\t\t" << getTuningAttributeName(getTuningAttributes(static_cast<VariantID>(j), t))
                         << std::endl;
    }
  }
  os << "\t\t\t its_per_rep = " << its_per_rep << std::endl;
  os << "\t\t\t kernels_per_rep = " << kernels_per_rep << std::endl;
  os << "\t\t\t bytes_allocated_per_rep = " << bytes_allocated_per_rep << std::endl;
  os << "\t\t\t bytes_read_per_rep = " << bytes_read_per_rep << std::endl;
  os << "\t\t\t bytes_written_per_rep = " << bytes_written_per_rep << std::endl;
  os << "\t\t\t bytes_modify_written_per_rep = " << bytes_modify_written_per_rep << std::endl;
  os << "\t\t\t bytes_atomic_modify_written_per_rep = " << bytes_atomic_modify_written_per_rep << std::endl;
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
  os << "\t\t\t checksum_reference_variant = " << getVariantName(checksum_reference_variant) << std::endl;
  os << "\t\t\t checksum_reference_tuning = " << checksum_reference_tuning << std::endl;
  os << "\t\t\t checksum_reference_tuning_attributes = " << getTuningAttributeName(checksum_reference_tuning_attributes) << std::endl;
  os << "\t\t\t checksum_reference = " << checksum_reference << std::endl;
  os << "\t\t\t checksum_min: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < checksum_min[j].size(); ++t) {
      os << "\t\t\t\t\t" << checksum_min[j][t] << std::endl;
    }
  }
  os << "\t\t\t checksum_max: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < checksum_max[j].size(); ++t) {
      os << "\t\t\t\t\t" << checksum_max[j][t] << std::endl;
    }
  }
  os << "\t\t\t checksum_sum: " << std::endl;
  for (unsigned j = 0; j < NumVariants; ++j) {
    os << "\t\t\t\t" << getVariantName(static_cast<VariantID>(j))
                     << " :" << std::endl;
    for (size_t t = 0; t < checksum_sum[j].size(); ++t) {
      os << "\t\t\t\t\t" << checksum_sum[j][t].get() << std::endl;
    }
  }
  os << std::endl;
}

#if defined(RAJA_PERFSUITE_USE_CALIPER)
void KernelBase::doOnceCaliMetaBegin(VariantID vid, size_t tune_idx)
{
  if(doCaliMetaOnce[vid].at(tune_idx)) {
    // Set values for Index_type.
    // Some of these may overflow if using "cali_set_int"
    auto cali_set_helper = [](cali_id_t const& attr, Index_type val) {
      cali_set(attr, &val, sizeof(Index_type));
    };
    cali_set_helper(ProblemSize_attr, getActualProblemSize());
    cali_set_helper(Reps_attr, getRunReps());
    cali_set_helper(Iters_Rep_attr, getItsPerRep());
    cali_set_helper(Kernels_Rep_attr, getKernelsPerRep());
    cali_set_helper(Bytes_Allocated_Rep_attr, getBytesAllocatedPerRep());
    cali_set_helper(Bytes_Moved_Rep_attr, getBytesMovedPerRep());
    cali_set_helper(Bytes_Touched_Rep_attr, getBytesTouchedPerRep());
    cali_set_helper(Bytes_Read_Rep_attr, getBytesReadPerRep());
    cali_set_helper(Bytes_Written_Rep_attr, getBytesWrittenPerRep());
    cali_set_helper(Bytes_ModifyWritten_Rep_attr, getBytesModifyWrittenPerRep());
    cali_set_helper(Bytes_AtomicModifyWritten_Rep_attr, getBytesAtomicModifyWrittenPerRep());
    cali_set_helper(Flops_Rep_attr, getFLOPsPerRep());
    cali_set_helper(BlockSize_attr, getBlockSize());
    cali_set_helper(MaxPerfectLoopDimensions_attr, getMaxPerfectLoopDimensions());
    cali_set_helper(ProblemDimensionality_attr, getProblemDimensionality());

    // Feature values will be either (0, 1)
    for (unsigned i = 0; i < FeatureID::NumFeatures; ++i) {
        FeatureID fid = static_cast<FeatureID>(i);
        std::string feature = getFeatureName(fid);
        cali_set_int(Feature_attrs[feature], usesFeature(fid));
    }

    cali_set_string(ChecksumConsistency_attr, getChecksumConsistencyName(getChecksumConsistency()).c_str());
    cali_set_string(Complexity_attr, getComplexityName(getComplexity()).c_str());
  }
}

void KernelBase::doOnceCaliMetaEnd(VariantID vid, size_t tune_idx)
{
  if(doCaliMetaOnce[vid].at(tune_idx)) {
    doCaliMetaOnce[vid].at(tune_idx) = false;
  }
}

void KernelBase::setCaliperMgrVariantTuning(VariantID vid,
                                  std::string tstr,
                                  const std::string& outfile,
                                  const std::string& addToSpotConfig,
                                  const std::string& addToCaliConfig,
                                  const int num_variants_tunings)
{
  static bool ran_spot_config_check = false;
  bool config_ok = true;

  const char* kernel_info_spec = R"json(
  {
    "name": "rajaperf_kernel_info",
    "type": "boolean",
    "category": "metric",
    "description": "Record RAJAPerf kernel info attributes",
    "query":
    [
      {
        "level"  : "local",
        "select" :
        [
          { "expr": "any(max#ProblemSize)", "as": "ProblemSize" },
          { "expr": "any(max#Reps)", "as": "Reps" },
          { "expr": "any(max#Iterations/Rep)", "as": "Iterations/Rep" },
          { "expr": "any(max#Kernels/Rep)", "as": "Kernels/Rep" },
          { "expr": "any(max#BytesMoved/Rep)", "as": "BytesMoved/Rep" },
          { "expr": "any(max#BytesTouched/Rep)", "as": "BytesTouched/Rep" },
          { "expr": "any(max#BytesRead/Rep)", "as": "BytesRead/Rep" },
          { "expr": "any(max#BytesWritten/Rep)", "as": "BytesWritten/Rep" },
          { "expr": "any(max#BytesModifyWritten/Rep)", "as": "BytesModifyWritten/Rep" },
          { "expr": "any(max#BytesAtomicModifyWritten/Rep)", "as": "BytesAtomicModifyWritten/Rep" },
          { "expr": "any(max#Flops/Rep)", "as": "Flops/Rep" },
          { "expr": "any(max#BlockSize)", "as": "BlockSize" },
          { "expr": "any(max#Forall)", "as": "FeatureForall" },
          { "expr": "any(max#Kernel)", "as": "FeatureKernel" },
          { "expr": "any(max#Launch)", "as": "FeatureLaunch" },
          { "expr": "any(max#Sort)", "as": "FeatureSort" },
          { "expr": "any(max#Scan)", "as": "FeatureScan" },
          { "expr": "any(max#Workgroup)", "as": "FeatureWorkgroup" },
          { "expr": "any(max#Reduction)", "as": "FeatureReduction" },
          { "expr": "any(max#Atomic)", "as": "FeatureAtomic" },
          { "expr": "any(max#View)", "as": "FeatureView" },
          { "expr": "any(max#MPI)", "as": "FeatureMPI" },
          { "expr": "any(max#MaxPerfectLoopDimensions)", "as": "MaxPerfectLoopDimensions" },
          { "expr": "any(max#ProblemDimensionality)", "as": "ProblemDimensionality" },
        ],
        "group by": ["Complexity", "ChecksumConsistency"],
      },
      {
        "level"  : "cross",
        "select" :
        [
          { "expr": "any(any#max#ProblemSize)", "as": "ProblemSize" },
          { "expr": "any(any#max#Reps)", "as": "Reps" },
          { "expr": "any(any#max#Iterations/Rep)", "as": "Iterations/Rep" },
          { "expr": "any(any#max#Kernels/Rep)", "as": "Kernels/Rep" },
          { "expr": "any(any#max#BytesMoved/Rep)", "as": "BytesMoved/Rep" },
          { "expr": "any(any#max#BytesTouched/Rep)", "as": "BytesTouched/Rep" },
          { "expr": "any(any#max#BytesRead/Rep)", "as": "BytesRead/Rep" },
          { "expr": "any(any#max#BytesWritten/Rep)", "as": "BytesWritten/Rep" },
          { "expr": "any(any#max#BytesModifyWritten/Rep)", "as": "BytesModifyWritten/Rep" },
          { "expr": "any(any#max#BytesAtomicModifyWritten/Rep)", "as": "BytesAtomicModifyWritten/Rep" },
          { "expr": "any(any#max#Flops/Rep)", "as": "Flops/Rep" },
          { "expr": "any(any#max#BlockSize)", "as": "BlockSize" },
          { "expr": "any(any#max#Forall)", "as": "FeatureForall" },
          { "expr": "any(any#max#Kernel)", "as": "FeatureKernel" },
          { "expr": "any(any#max#Launch)", "as": "FeatureLaunch" },
          { "expr": "any(any#max#Sort)", "as": "FeatureSort" },
          { "expr": "any(any#max#Scan)", "as": "FeatureScan" },
          { "expr": "any(any#max#Workgroup)", "as": "FeatureWorkgroup" },
          { "expr": "any(any#max#Reduction)", "as": "FeatureReduction" },
          { "expr": "any(any#max#Atomic)", "as": "FeatureAtomic" },
          { "expr": "any(any#max#View)", "as": "FeatureView" },
          { "expr": "any(any#max#MPI)", "as": "FeatureMPI" },
          { "expr": "any(any#max#MaxPerfectLoopDimensions)", "as": "MaxPerfectLoopDimensions" },
          { "expr": "any(any#max#ProblemDimensionality)", "as": "ProblemDimensionality" },
        ],
        "group by": ["Complexity", "ChecksumConsistency"],
      }
    ]
  }
  )json";

  // Update these later if CALI_CONFIG present
  std::string updatedSpotConfig = addToSpotConfig;
  std::string updatedCaliConfig = addToCaliConfig;

  // Parse CALI_CONFIG if provided
  const char* cali_config_env = std::getenv("DISABLED_CALI_CONFIG");
  if (cali_config_env) {
    std::string cali_config(cali_config_env);
    std::cout << "CALI_CONFIG: " << cali_config << std::endl;

    // Match spot() config
    std::regex pattern(R"(spot\(([^)]*)\))");
    std::smatch match;
    if (std::regex_search(cali_config, match, pattern)) {
      std::string spot_config = match[1];
      std::regex file_pattern(R"(\boutput=[^,)]*\.cali)");

      // Remove cali file from config
      std::string withoutFile = std::regex_replace(spot_config, file_pattern, "");
      std::smatch file_match;
      if (std::regex_search(spot_config, file_match, file_pattern)) {
        std::cout << "WARNING: Removing requested output name from config: '"
                  << file_match[0]
                  << "'. Output cali file name will be automatically generated."
                  << std::endl;
      }

      updatedSpotConfig += withoutFile;
    }

    // Parameters outside the spot config directly added to cali config
    std::string remaining_config = std::regex_replace(cali_config, pattern, "");
    updatedCaliConfig += remaining_config;

    auto trimCommas = [](std::string &str) {
      if (!str.empty() && str.front() == ',')
        str.erase(0, 1);
      if (!str.empty() && str.back() == ',')
        str.pop_back();
    };
    trimCommas(updatedCaliConfig);
    trimCommas(updatedSpotConfig);
  }

  // Caliper configuration check. Skip check if both empty
  if ((!updatedSpotConfig.empty() || !updatedCaliConfig.empty()) && !ran_spot_config_check) {
    cali::ConfigManager cm;
    std::string check_profile;
    // If both not empty
    if (!updatedSpotConfig.empty() && !updatedCaliConfig.empty()) {
      check_profile = updatedCaliConfig + ",spot(" + updatedSpotConfig + ")";
    }
    else if (!updatedSpotConfig.empty()) {
      check_profile = "spot(" + updatedSpotConfig + ")";
    }
    // if !updatedCaliConfig.empty()
    else {
      check_profile = updatedCaliConfig;
    }

    std::string msg = cm.check(check_profile.c_str());
    if(!msg.empty()) {
      std::cerr << "Problem with Cali Config: " << check_profile << "\n";
      std::cerr << msg << "\n";
      config_ok = false;
      exit(-1);
    }
    ran_spot_config_check = true;
    std::cout << "Caliper ran Spot config check\n";
  }

  // Setup variant/tuning caliper config if check passes
  if(config_ok) {
    cali::ConfigManager m;
    mgr[vid][tstr] = m;
    std::string vstr = getVariantName(vid);
    std::string profile = "";
    std::string sprofile = "";
    std::string ccprofile = "";
    if (!updatedCaliConfig.empty()) {
      ccprofile += updatedCaliConfig + ",";
    }
    // If --outfile not provided, give generic name
    if (outfile == "RAJAPerf") {
      sprofile = "spot(output=" + vstr + "-" + tstr + ".cali";
    }
    else {
      // Ensure cali files for each variant/tuning are not same file name
      if (num_variants_tunings > 1)
        throw std::runtime_error("Error: Cannot use '--outfile' with Caliper if running multiple variants/tunings. Must be running single variant & tuning.");
      sprofile = "spot(output=" + outfile + ".cali";
    }
    if(!updatedSpotConfig.empty()) {
      sprofile += "," + updatedSpotConfig;
    }
    sprofile += ")";
    profile = ccprofile + sprofile;
    std::cout << "Profile: " << profile << std::endl;
    mgr[vid][tstr].add_option_spec(kernel_info_spec);
    mgr[vid][tstr].set_default_parameter("rajaperf_kernel_info", "true");
    mgr[vid][tstr].add(profile.c_str());
  }
}

// initialize a KernelBase static
std::map<rajaperf::VariantID, std::map<std::string, cali::ConfigManager>> KernelBase::mgr;
#endif
}  // closing brace for rajaperf namespace
