//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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

#ifdef RAJAPERF_USE_CALIPER

#define CALI_START \
    if(doCaliperTiming) { \
      std::string kstr = getName(); \
      std::string gstr = getGroupName(kstr); \
      std::string vstr = getVariantName(running_variant); \
      CALI_MARK_BEGIN(vstr.c_str()); \
      CALI_MARK_BEGIN(gstr.c_str()); \
      CALI_MARK_BEGIN(kstr.c_str()); \
    }

#define CALI_STOP \
    if(doCaliperTiming) { \
      std::string kstr = getName(); \
      std::string gstr = getGroupName(kstr); \
      std::string vstr = getVariantName(running_variant); \
      CALI_MARK_END(kstr.c_str()); \
      CALI_MARK_END(gstr.c_str()); \
      CALI_MARK_END(vstr.c_str()); \
    }
#else
#define CALI_START
#define CALI_STOP
#endif

#include <string>
#include <iostream>
#include <map>

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

  SizeSpec getSizeSpec() {return run_params.getSizeSpec();}

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

  bool hasVariantToRun(VariantID vid) const { return has_variant_to_run[vid]; }

  void setVariantDefined(VariantID vid);

  void execute(VariantID vid);

#ifdef RAJAPERF_USE_CALIPER
  void caliperOn() { doCaliperTiming = true; }
  void caliperOff() { doCaliperTiming = false; } 
  static void setCaliperMgrVariant(VariantID vid)
  {
    cali::ConfigManager m;
    mgr.insert(std::make_pair(vid,m));
    std::string vstr = getVariantName(vid);
    std::string profile = "spot(output=" + vstr + ".cali)";
    std::cout << "Profile: " << profile << std::endl;
    mgr[vid].add(profile.c_str()); 
  }

  static void setCaliperMgrStart(VariantID vid) { mgr[vid].start(); }
  static void setCaliperMgrStop(VariantID vid) { mgr[vid].stop(); }
  static void setCaliperMgrFlush() 
  { // we're going to flush all the variants at once
    for(auto const &kv : mgr) {
      // set Adiak key first
      std::string variant=getVariantName(kv.first);
      adiak::value("variant",variant.c_str());
      mgr[kv.first].flush(); 
    }
  }

  std::string getGroupName(const std::string &kname )
  {
    std::size_t found = kname.find("_");
    return kname.substr(0,found);
  }

#endif

  void startTimer() 
  { 
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA || running_variant == RAJA_CUDA ) {
      cudaDeviceSynchronize();
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    if ( running_variant == Base_HIP || running_variant == RAJA_HIP ) {
      hipDeviceSynchronize();
    }
#endif
    CALI_START;
    timer.start(); 
  }

  void stopTimer()  
  { 
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA || running_variant == RAJA_CUDA ) {
      cudaDeviceSynchronize();
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    if ( running_variant == Base_HIP || running_variant == RAJA_HIP ) {
      hipDeviceSynchronize();
    }
#endif
    timer.stop(); 
    CALI_STOP;
    recordExecTime(); 
  }

  void resetTimer() { timer.reset(); }

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by each concrete kernel class.
  //

  virtual Index_type getItsPerRep() const { return getRunSize(); }

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

protected:
  const RunParams& run_params;

  Checksum_type checksum[NumVariants];

private:
  KernelBase() = delete;

  void recordExecTime(); 

  KernelID    kernel_id;
  std::string name;

  Index_type default_size;
  Index_type default_reps;

  VariantID running_variant; 

  int num_exec[NumVariants];

  RAJA::Timer timer;

  RAJA::Timer::ElapsedType min_time[NumVariants];
  RAJA::Timer::ElapsedType max_time[NumVariants];
  RAJA::Timer::ElapsedType tot_time[NumVariants];

  bool has_variant_to_run[NumVariants];

#ifdef RAJAPERF_USE_CALIPER
  bool doCaliperTiming = true; // warmup can use this to exclude timing
// we need a Caliper Manager object per variant
  static std::map<rajaperf::VariantID, cali::ConfigManager> mgr;
#endif
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
