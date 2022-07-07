//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_KernelBase_HPP
#define RAJAPerf_KernelBase_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RPTypes.hpp"
#include "common/DataUtils.hpp"
#include "common/RunParams.hpp"
#include "common/GPUUtils.hpp"

#include "RAJA/util/Timer.hpp"
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include <mpi.h>
#endif
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif

#include <string>
#include <vector>
#include <iostream>
#include <limits>

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
  static constexpr size_t getUnknownTuningIdx()
    { return std::numeric_limits<size_t>::max(); }
  static std::string getDefaultTuningName() { return "default"; }

  KernelBase(KernelID kid, const RunParams& params);

  virtual ~KernelBase();

  KernelID     getKernelID() const { return kernel_id; }
  const std::string& getName() const { return name; }

  //
  // Methods called in kernel subclass constructors to set kernel
  // properties used to describe kernel and define how it will run
  //

  void setDefaultProblemSize(Index_type size) { default_prob_size = size; }
  void setActualProblemSize(Index_type size) { actual_prob_size = size; }
  void setDefaultReps(Index_type reps) { default_reps = reps; }
  void setItsPerRep(Index_type its) { its_per_rep = its; };
  void setKernelsPerRep(Index_type nkerns) { kernels_per_rep = nkerns; };
  void setBytesPerRep(Index_type bytes) { bytes_per_rep = bytes;}
  void setFLOPsPerRep(Index_type FLOPs) { FLOPs_per_rep = FLOPs; }

  void setUsesFeature(FeatureID fid) { uses_feature[fid] = true; }
  void setVariantDefined(VariantID vid);
  void addVariantTuningName(VariantID vid, std::string name)
  { variant_tuning_names[vid].emplace_back(std::move(name)); }

  virtual void setSeqTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  virtual void setOpenMPTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#endif
#if defined(RAJA_ENABLE_CUDA)
  virtual void setCudaTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#endif
#if defined(RAJA_ENABLE_HIP)
  virtual void setHipTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  virtual void setOpenMPTargetTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#endif
#if defined(RUN_KOKKOS)
  virtual void setKokkosTuningDefinitions(VariantID vid)
  { addVariantTuningName(vid, getDefaultTuningName()); }
#endif

  //
  // Getter methods used to generate kernel execution summary
  // and kernel details report ouput.
  //

  Index_type getDefaultProblemSize() const { return default_prob_size; }
  Index_type getActualProblemSize() const { return actual_prob_size; }
  Index_type getDefaultReps() const { return default_reps; }
  Index_type getItsPerRep() const { return its_per_rep; };
  Index_type getKernelsPerRep() const { return kernels_per_rep; };
  Index_type getBytesPerRep() const { return bytes_per_rep; }
  Index_type getFLOPsPerRep() const { return FLOPs_per_rep; }

  Index_type getTargetProblemSize() const;
  Index_type getRunReps() const;

  bool usesFeature(FeatureID fid) const { return uses_feature[fid]; };

  bool hasVariantDefined(VariantID vid) const
    { return !variant_tuning_names[vid].empty(); }
  bool hasVariantTuningDefined(VariantID vid, size_t tune_idx) const
    {
      if (hasVariantDefined(vid) && tune_idx < getNumVariantTunings(vid)) {
        return true;
      }
      return false;
    }
  bool hasVariantTuningDefined(VariantID vid, std::string const& tuning_name) const
    {
      if (hasVariantDefined(vid)) {
        for (std::string const& a_tuning_name : getVariantTuningNames(vid)) {
          if (tuning_name == a_tuning_name) { return true; }
        }
      }
      return false;
    }
  size_t getVariantTuningIndex(VariantID vid, std::string const& tuning_name) const
    {
      std::vector<std::string> const& tuning_names = getVariantTuningNames(vid);
      for (size_t t = 0; t < tuning_names.size(); ++t) {
        std::string const& a_tuning_name = tuning_names[t];
        if (tuning_name == a_tuning_name) { return t; }
      }
      return getUnknownTuningIdx();
    }
  size_t getNumVariantTunings(VariantID vid) const
    { return getVariantTuningNames(vid).size(); }
  std::string const& getVariantTuningName(VariantID vid, size_t tune_idx) const
    { return getVariantTuningNames(vid).at(tune_idx); }
  std::vector<std::string> const& getVariantTuningNames(VariantID vid) const
    { return variant_tuning_names[vid]; }

  //
  // Methods to get information about kernel execution for reports
  // containing kernel execution information
  //
  bool wasVariantTuningRun(VariantID vid, size_t tune_idx) const
    {
      if (tune_idx != getUnknownTuningIdx()) {
        return num_exec[vid].at(tune_idx) > 0;
      }
      return false;
    }

  // get runtime of executed variant/tuning
  double getLastTime() const { return timer.elapsed(); }

  // get timers accumulated over npasses
  double getMinTime(VariantID vid, size_t tune_idx) const { return min_time[vid].at(tune_idx); }
  double getMaxTime(VariantID vid, size_t tune_idx) const { return max_time[vid].at(tune_idx); }
  double getTotTime(VariantID vid, size_t tune_idx) { return tot_time[vid].at(tune_idx); }
  Checksum_type getChecksum(VariantID vid, size_t tune_idx) const { return checksum[vid].at(tune_idx); }

  void execute(VariantID vid, size_t tune_idx);

  void synchronize()
  {
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA ||
         running_variant == Lambda_CUDA ||
         running_variant == RAJA_CUDA ) {
      cudaErrchk( cudaDeviceSynchronize() );
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    if ( running_variant == Base_HIP ||
         running_variant == Lambda_HIP ||
         running_variant == RAJA_HIP ) {
      hipErrchk( hipDeviceSynchronize() );
    }
#endif
  }

  void startTimer()
  {
    synchronize();
#ifdef RAJA_PERFSUITE_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timer.start();
  }

  void stopTimer()
  {
    synchronize();
#ifdef RAJA_PERFSUITE_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timer.stop(); recordExecTime();
  }

  void resetTimer() { timer.reset(); }

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by concrete kernel subclass.
  //

  virtual void print(std::ostream& os) const;

  virtual void runKernel(VariantID vid, size_t tune_idx);

  virtual void setUp(VariantID vid, size_t tune_idx) = 0;
  virtual void updateChecksum(VariantID vid, size_t tune_idx) = 0;
  virtual void tearDown(VariantID vid, size_t tune_idx) = 0;

  virtual void runSeqVariant(VariantID vid, size_t tune_idx) = 0;
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  virtual void runOpenMPVariant(VariantID vid, size_t tune_idx) = 0;
#endif
#if defined(RAJA_ENABLE_CUDA)
  virtual void runCudaVariant(VariantID vid, size_t tune_idx) = 0;
#endif
#if defined(RAJA_ENABLE_HIP)
  virtual void runHipVariant(VariantID vid, size_t tune_idx) = 0;
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  virtual void runOpenMPTargetVariant(VariantID vid, size_t tune_idx) = 0;
#endif
  virtual void runStdParVariant(VariantID vid, size_t tune_idx) = 0;
#if defined(RUN_KOKKOS)
  virtual void runKokkosVariant(VariantID vid, size_t tune_idx)
  {
     getCout() << "\n KernelBase: Unimplemented Kokkos variant id = " << vid << std::endl;
  }
#endif

protected:
  const RunParams& run_params;

  std::vector<Checksum_type> checksum[NumVariants];
  Checksum_type checksum_scale_factor;

private:
  KernelBase() = delete;

  void recordExecTime();

  //
  // Static properties of kernel, independent of run
  //
  KernelID    kernel_id;
  std::string name;

  Index_type default_prob_size;
  Index_type default_reps;

  Index_type actual_prob_size;

  bool uses_feature[NumFeatures];

  std::vector<std::string> variant_tuning_names[NumVariants];

  //
  // Properties of kernel dependent on how kernel is run
  //
  Index_type its_per_rep;
  Index_type kernels_per_rep;
  Index_type bytes_per_rep;
  Index_type FLOPs_per_rep;

  VariantID running_variant;
  size_t running_tuning;

  std::vector<int> num_exec[NumVariants];

  RAJA::Timer timer;

  std::vector<RAJA::Timer::ElapsedType> min_time[NumVariants];
  std::vector<RAJA::Timer::ElapsedType> max_time[NumVariants];
  std::vector<RAJA::Timer::ElapsedType> tot_time[NumVariants];
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
