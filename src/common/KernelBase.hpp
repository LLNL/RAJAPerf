//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
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

#include "RAJA/util/Timer.hpp"
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif

#include <string>
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
    { return has_variant_defined[vid]; }


  //
  // Methods to get information about kernel execution for reports
  // containing kernel execution information
  //
  bool wasVariantRun(VariantID vid) const
    { return num_exec[vid] > 0; }

  double getMinTime(VariantID vid) const { return min_time[vid]; }
  double getMaxTime(VariantID vid) const { return max_time[vid]; }
  double getTotTime(VariantID vid) { return tot_time[vid]; }
  Checksum_type getChecksum(VariantID vid) const { return checksum[vid]; }

  void execute(VariantID vid);

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
    timer.start();
  }

  void stopTimer()
  {
    synchronize();
    timer.stop(); recordExecTime();
  }

  void resetTimer() { timer.reset(); }

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by concrete kernel subclass.
  //

  virtual void print(std::ostream& os) const;

  virtual void runKernel(VariantID vid);

  virtual void setUp(VariantID vid) = 0;
  virtual void updateChecksum(VariantID vid) = 0;
  virtual void tearDown(VariantID vid) = 0;

  virtual void runSeqVariant(VariantID vid) = 0;
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  virtual void runOpenMPVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_CUDA)
  virtual void runCudaVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_HIP)
  virtual void runHipVariant(VariantID vid) = 0;
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  virtual void runOpenMPTargetVariant(VariantID vid) = 0;
#endif
  virtual void runStdParVariant(VariantID vid) = 0;

protected:
  const RunParams& run_params;

  Checksum_type checksum[NumVariants];
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

  bool has_variant_defined[NumVariants];

  //
  // Properties of kernel dependent on how kernel is run
  //
  Index_type its_per_rep;
  Index_type kernels_per_rep;
  Index_type bytes_per_rep;
  Index_type FLOPs_per_rep;

  VariantID running_variant;

  int num_exec[NumVariants];

  RAJA::Timer timer;

  RAJA::Timer::ElapsedType min_time[NumVariants];
  RAJA::Timer::ElapsedType max_time[NumVariants];
  RAJA::Timer::ElapsedType tot_time[NumVariants];
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
