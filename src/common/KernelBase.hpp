//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
#include "RAJA/util/reduce.hpp"
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include <mpi.h>
#endif
#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif
#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#endif
#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/util/sycl_compat.hpp"
#endif

#include "camp/resource.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <limits>
#include <utility>

#if defined(RAJA_PERFSUITE_USE_CALIPER)

#define CALI_START \
    if (doCaliperTiming) { \
      std::string kstr = getName(); \
      std::string gstr = getKernelGroupName(kstr); \
      std::string vstr = "RAJAPerf"; \
      doOnceCaliMetaBegin(running_variant, running_tuning); \
      CALI_MARK_BEGIN(vstr.c_str()); \
      CALI_MARK_BEGIN(gstr.c_str()); \
      CALI_MARK_BEGIN(kstr.c_str()); \
    }

#define CALI_STOP \
    if (doCaliperTiming) { \
      std::string kstr = getName(); \
      std::string gstr = getKernelGroupName(kstr); \
      std::string vstr = "RAJAPerf"; \
      CALI_MARK_END(kstr.c_str()); \
      CALI_MARK_END(gstr.c_str()); \
      CALI_MARK_END(vstr.c_str()); \
      doOnceCaliMetaEnd(running_variant,running_tuning); \
    }

#else

#define CALI_START
#define CALI_STOP

#endif

#if defined(RAJA_PERFSUITE_USE_CALIPER) && \
    defined(RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL)

#define RP_CALI_SUBKERNEL_BEGIN(name) \
    if (isCaliperTiming()) { \
      cali_begin_string(Subkernel_attr, name); \
    }

#define RP_CALI_SUBKERNEL_END(name) \
    if (isCaliperTiming()) { \
      cali_end(Subkernel_attr); \
    }

#else

#define RP_CALI_SUBKERNEL_BEGIN(name)
#define RP_CALI_SUBKERNEL_END(name)

#endif

//
// Macro to increment rep loop counter: quiets C++20 compiler warning
//
#define RP_REPCOUNTINC(var)  static_cast<void>( ((var = var + 1), 0) )

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

  //
  // Method to set state of all Kernel objects to indicate kernel runs 
  // are for warmup purposes if true is passed, else false.
  //
  // The warmup state before the method call is returned to facilitate 
  // reset mechanics. 
  //
  static bool setWarmupRun(bool warmup_run);

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
  void setRunReps(Index_type reps) { actual_reps = reps; }
  void setItsPerRep(Index_type its) { its_per_rep = its; };
  void setKernelsPerRep(Index_type nkerns) { kernels_per_rep = nkerns; };
  void setBytesAllocatedPerRep(Index_type bytes) { bytes_allocated_per_rep = bytes;}
  void setBytesReadPerRep(Index_type bytes) { bytes_read_per_rep = bytes;}
  void setBytesWrittenPerRep(Index_type bytes) { bytes_written_per_rep = bytes;}
  void setBytesModifyWrittenPerRep(Index_type bytes) { bytes_modify_written_per_rep = bytes;}
  void setBytesAtomicModifyWrittenPerRep(Index_type bytes) { bytes_atomic_modify_written_per_rep = bytes;}
  void setFLOPsPerRep(Index_type FLOPs) { FLOPs_per_rep = FLOPs; }
  void setBlockSize(Index_type size) { kernel_block_size = size; }
  void setChecksumConsistency(ChecksumConsistency cc) { checksum_consistency = cc; }
  void setChecksumTolerance(Checksum_type ct) { checksum_tolerance = ct; }
  void setComplexity(Complexity ac) { complexity = ac; }
  void setMaxPerfectLoopDimensions(Index_type nploops) { num_nested_perfect_loops = nploops; }
  void setProblemDimensionality(Index_type pdim) { problem_dimensionality = pdim; }

  void setUsesFeature(FeatureID fid) { uses_feature[fid] = true; }

  virtual void defineSeqVariantTunings() {}

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  virtual void defineOpenMPVariantTunings() {}
#endif

#if defined(RAJA_ENABLE_CUDA)
  virtual void defineCudaVariantTunings() {}
#endif

#if defined(RAJA_ENABLE_HIP)
  virtual void defineHipVariantTunings() {}
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  virtual void defineOpenMPTargetVariantTunings() {}
#endif

#if defined(RUN_KOKKOS)
  virtual void defineKokkosVariantTunings() {}
#endif

#if defined(RAJA_ENABLE_SYCL)
  virtual void defineSyclVariantTunings() {}
#endif

  template < auto method >
  void addVariantTuning(VariantID vid, std::string name,
                        TuningAttribute attrs = TuningAttribute::none)
  {
    addVariantTuning(vid, std::move(name), attrs,
        &KernelBase::wrapDerivedVariantTuningMethod<
            class_of_member_function_pointer_t<decltype(method)>, method>);
  }

  void addVariantTunings()
  {
    defineSeqVariantTunings();

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    defineOpenMPVariantTunings();
#endif

#if defined(RAJA_ENABLE_CUDA)
    defineCudaVariantTunings();
#endif

#if defined(RAJA_ENABLE_HIP)
    defineHipVariantTunings();
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    defineOpenMPTargetVariantTunings();
#endif

#if defined(RUN_KOKKOS)
    defineKokkosVariantTunings();
#endif

#if defined(RAJA_ENABLE_SYCL)
    defineSyclVariantTunings();
#endif
  }


  //
  // Getter methods used to generate kernel execution summary
  // and kernel details report ouput.
  //

  Index_type getDefaultProblemSize() const { return default_prob_size; }
  Index_type getActualProblemSize() const { return actual_prob_size; }
  Index_type getDefaultReps() const { return default_reps; }
  Index_type getRunReps() const { return s_warmup_run ? 1 : actual_reps; }
  Index_type getItsPerRep() const { return its_per_rep; }
  Index_type getKernelsPerRep() const { return kernels_per_rep; }
  Index_type getBytesAllocatedPerRep() const { return bytes_allocated_per_rep; }
  Index_type getBytesMovedPerRep() const { return bytes_read_per_rep + bytes_written_per_rep + 2*bytes_modify_written_per_rep + 2*bytes_atomic_modify_written_per_rep; } // count modify_write operations twice to get the memory traffic
  Index_type getBytesTouchedPerRep() const { return bytes_read_per_rep + bytes_written_per_rep + bytes_modify_written_per_rep + bytes_atomic_modify_written_per_rep; } // count modify_write operations once to get the data size only
  Index_type getBytesReadPerRep() const { return bytes_read_per_rep + bytes_modify_written_per_rep; }
  Index_type getBytesWrittenPerRep() const { return bytes_written_per_rep + bytes_modify_written_per_rep; }
  Index_type getBytesModifyWrittenPerRep() const { return bytes_modify_written_per_rep; }
  Index_type getBytesAtomicModifyWrittenPerRep() const { return bytes_atomic_modify_written_per_rep; }
  Index_type getFLOPsPerRep() const { return FLOPs_per_rep; }
  double getBlockSize() const { return kernel_block_size; }
  ChecksumConsistency getChecksumConsistency() const { return checksum_consistency; };
  Checksum_type getChecksumTolerance() const { return checksum_tolerance; }
  Complexity getComplexity() const { return complexity; };
  Index_type getMaxPerfectLoopDimensions() const { return num_nested_perfect_loops; };
  Index_type getProblemDimensionality() const { return problem_dimensionality; };


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
  bool hasVariantTuningDefined(VariantID vid,
                               std::string const& tuning_name) const
  {
    return getVariantTuningIndex(vid, tuning_name) != getUnknownTuningIdx();
  }

  size_t getVariantTuningIndex(VariantID vid,
                               std::string const& tuning_name) const
  {
    if (hasVariantDefined(vid)) {
      std::vector<std::string> const& tuning_names = getVariantTuningNames(vid);
      for (size_t t = 0; t < tuning_names.size(); ++t) {
        if (tuning_name == tuning_names[t]) {
          return t;
        }
      }
    }
    return getUnknownTuningIdx();
  }

  size_t getNumVariantTunings(VariantID vid) const
  { return getVariantTuningNames(vid).size(); }
  std::string const& getVariantTuningName(VariantID vid, size_t tune_idx) const
  { return getVariantTuningNames(vid).at(tune_idx); }
  std::vector<std::string> const& getVariantTuningNames(VariantID vid) const
  { return variant_tuning_names[vid]; }

  TuningAttribute getTuningAttributes(VariantID vid, size_t tune_idx) const
  { return variant_tuning_attrs[vid].at(tune_idx); }

  //
  // Methods to get information about kernel execution for reports
  // containing kernel execution information
  //
  bool wasVariantTuningRun(VariantID vid, size_t tune_idx) const
  {
    if (tune_idx != getUnknownTuningIdx() && hasVariantDefined(vid)) {
      return num_exec[vid].at(tune_idx) > 0;
    }
    return false;
  }
  ///
  bool wasVariantTuningRun(VariantID vid, std::string const& tuning_name) const
  {
    size_t tune_idx = getVariantTuningIndex(vid, tuning_name);
    return wasVariantTuningRun(vid, tune_idx) ;
  }

  // get runtime of executed variant/tuning
  double getLastTime() const { return timer.elapsed(); }

  // get timers accumulated over npasses
  double getMinTime(VariantID vid, size_t tune_idx) const
  { return min_time[vid].at(tune_idx); }
  double getMaxTime(VariantID vid, size_t tune_idx) const
  { return max_time[vid].at(tune_idx); }
  double getTotTime(VariantID vid, size_t tune_idx) const
  { return tot_time[vid].at(tune_idx); }

  // Get reference checksum (first variant tuning run)
  Checksum_type getReferenceChecksum() const
  {
    if (checksum_reference_variant == NumVariants) {
      throw std::runtime_error("Can't get reference checksum average if kernel was not run");
    }
    return checksum_reference;
  }
  Checksum_type getLastChecksum() const
  {
    return checksum.get();
  }
  Checksum_type getChecksumAverage(VariantID vid, size_t tune_idx) const
  {
    if (num_exec[vid].at(tune_idx) <= 0) {
      throw std::runtime_error("Can't get checksum average if variant tuning was not run");
    }
    return checksum_sum[vid].at(tune_idx).get() / num_exec[vid].at(tune_idx);
  }
  static Checksum_type calculateChecksumRelativeAbsoluteDifference(
      Checksum_type checksum, Checksum_type reference_checksum)
  {
    Checksum_type checksum_abs_diff = std::abs(reference_checksum - checksum);

    Checksum_type checksum_rel_abs_diff =
        (reference_checksum == static_cast<Checksum_type>(0))
        ? checksum_abs_diff // handle case where checksum is 0 (Basic_EMPTY)
        : std::abs(checksum_abs_diff / reference_checksum) ;

    return checksum_rel_abs_diff;
  }
  Checksum_type getChecksumMaxRelativeAbsoluteDifference(VariantID vid, size_t tune_idx) const
  {
    if (num_exec[vid].at(tune_idx) <= 0) {
      throw std::runtime_error("Can't get checksum max rel abs diff if variant tuning was not run");
    }

    Checksum_type reference_checksum = getReferenceChecksum();

    Checksum_type cksum_max_rel_abs_diff =
        std::max( calculateChecksumRelativeAbsoluteDifference(
                      checksum_min[vid].at(tune_idx), reference_checksum),
                  calculateChecksumRelativeAbsoluteDifference(
                      checksum_max[vid].at(tune_idx), reference_checksum) );

    return cksum_max_rel_abs_diff;
  }

  void execute(VariantID vid, size_t tune_idx);

  camp::resources::Host getHostResource()
  {
    return camp::resources::Host::get_default();
  }

#if defined(RAJA_ENABLE_CUDA)
  camp::resources::Cuda getCudaResource()
  {
    if (run_params.getGPUStream() == 0) {
      return camp::resources::Cuda::CudaFromStream(0);
    }
    return camp::resources::Cuda::get_default();
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  camp::resources::Hip getHipResource()
  {
    if (run_params.getGPUStream() == 0) {
      return camp::resources::Hip::HipFromStream(0);
    }
    return camp::resources::Hip::get_default();
  }
#endif

#if defined(RAJA_ENABLE_SYCL)
  camp::resources::Sycl getSyclResource()
  {
    /*
    if (run_params.getGPUStream() == 0) {
      return camp::resources::Sycl::SyclFromStream(0);
    }
    */
    return camp::resources::Sycl::get_default();
  }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  camp::resources::Omp getOmpTargetResource()
  {
    return camp::resources::Omp::get_default();
  }
#endif

  void synchronize()
  {
#if defined(RAJA_ENABLE_CUDA)
    if ( running_variant == Base_CUDA ||
         running_variant == Lambda_CUDA ||
         running_variant == RAJA_CUDA ) {
      CAMP_CUDA_API_INVOKE_AND_CHECK( cudaDeviceSynchronize );
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    if ( running_variant == Base_HIP ||
         running_variant == Lambda_HIP ||
         running_variant == RAJA_HIP ) {
      CAMP_HIP_API_INVOKE_AND_CHECK( hipDeviceSynchronize );
    }
#endif
#if defined(RAJA_ENABLE_SYCL)
    if ( running_variant == Base_SYCL ||
         running_variant == RAJA_SYCL ) {
      getSyclResource().get_queue().wait();
    }
#endif

  }

  Size_type getDataAlignment() const;
  Size_type getNBytesPaddedToDataAlignment(Size_type size) const;

  template <typename T>
  T offsetPointer(T ptr, Size_type num_bytes) const
  {
    return reinterpret_cast<T>(reinterpret_cast<char*>(ptr) + num_bytes);
  }

  DataSpace getDataSpace(VariantID vid) const;
  DataSpace getReductionDataSpace(VariantID vid) const;
  DataSpace getMPIDataSpace(VariantID vid) const;

  template <typename T>
  void allocData(DataSpace dataSpace, T& ptr, Size_type len)
  {
    rajaperf::allocData(dataSpace,
        ptr, len, getDataAlignment());
  }

  template <typename T>
  void allocAndInitData(DataSpace dataSpace, T*& ptr, Size_type len)
  {
    rajaperf::allocAndInitData(dataSpace,
        ptr, len, getDataAlignment());
  }

  template <typename T, typename V>
  void allocAndInitDataConst(DataSpace dataSpace, T*& ptr, Size_type len, V val)
  {
    rajaperf::allocAndInitDataConst(dataSpace,
        ptr, len, getDataAlignment(), val);
  }

  template <typename T>
  void allocAndInitDataRandSign(DataSpace dataSpace, T*& ptr, Size_type len)
  {
    rajaperf::allocAndInitDataRandSign(dataSpace,
        ptr, len, getDataAlignment());
  }

  template <typename T>
  void allocAndInitDataRandValue(DataSpace dataSpace, T*& ptr, Size_type len)
  {
    rajaperf::allocAndInitDataRandValue(dataSpace,
        ptr, len, getDataAlignment());
  }

  template <typename T>
  rajaperf::AutoDataMover<T> scopedMoveData(DataSpace dataSpace, T*& ptr, Size_type len)
  {
    DataSpace hds = rajaperf::hostCopyDataSpace(dataSpace);
    rajaperf::moveData(hds, dataSpace, ptr, len, getDataAlignment());
    return {dataSpace, hds, ptr, len, getDataAlignment()};
  }

  template <typename T>
  void copyData(DataSpace dst_dataSpace, T* dst_ptr,
                DataSpace src_dataSpace, const T* src_ptr,
                Size_type len)
  {
    rajaperf::copyData(dst_dataSpace, dst_ptr, src_dataSpace, src_ptr, len);
  }

  template <typename T>
  void deallocData(DataSpace dataSpace, T& ptr)
  {
    rajaperf::deallocData(dataSpace, ptr);
  }

  template <typename T>
  void allocData(T*& ptr, Size_type len, VariantID vid)
  {
    rajaperf::allocData(getDataSpace(vid),
        ptr, len, getDataAlignment());
  }

  template <typename T>
  void allocAndCopyHostData(T*& dst_ptr,
                            const T* src_ptr,
                            Size_type len,
                            VariantID vid)
  {
    rajaperf::allocData(getDataSpace(vid),
        dst_ptr, len, getDataAlignment());

    rajaperf::copyData(getDataSpace(vid),
        dst_ptr, DataSpace::Host, src_ptr, len);
  }

  template <typename T>
  void allocAndInitData(T*& ptr, Size_type len, VariantID vid)
  {
    rajaperf::allocAndInitData(getDataSpace(vid),
        ptr, len, getDataAlignment());
  }

  template <typename T, typename V>
  void allocAndInitDataConst(T*& ptr, Size_type len, V val, VariantID vid)
  {
    rajaperf::allocAndInitDataConst(getDataSpace(vid),
        ptr, len, getDataAlignment(), val);
  }

  template <typename T>
  void allocAndInitDataRandSign(T*& ptr, Size_type len, VariantID vid)
  {
    rajaperf::allocAndInitDataRandSign(getDataSpace(vid),
        ptr, len, getDataAlignment());
  }

  template <typename T>
  void allocAndInitDataRandValue(T*& ptr, Size_type len, VariantID vid)
  {
    rajaperf::allocAndInitDataRandValue(getDataSpace(vid),
        ptr, len, getDataAlignment());
  }

  template <typename T>
  rajaperf::AutoDataMover<T> allocDataForInit(T*& ptr, Size_type len, VariantID vid)
  {
    DataSpace ds = getDataSpace(vid);
    DataSpace hds = rajaperf::hostCopyDataSpace(ds);
    rajaperf::allocData(hds, ptr, len, getDataAlignment());
    return {ds, hds, ptr, len, getDataAlignment()};
  }

  template <typename T>
  rajaperf::AutoDataMover<T> allocAndInitDataForInit(T*& ptr, Size_type len, VariantID vid)
  {
    DataSpace ds = getDataSpace(vid);
    DataSpace hds = rajaperf::hostCopyDataSpace(ds);
    rajaperf::allocAndInitData(hds, ptr, len, getDataAlignment());
    return {ds, hds, ptr, len, getDataAlignment()};
  }

  template <typename T>
  rajaperf::AutoDataMover<T> allocAndInitDataConstForInit(T*& ptr, Size_type len, T val, VariantID vid)
  {
    DataSpace ds = getDataSpace(vid);
    DataSpace hds = rajaperf::hostCopyDataSpace(ds);
    rajaperf::allocAndInitDataConst(hds, ptr, len, getDataAlignment(), val);
    return {ds, hds, ptr, len, getDataAlignment()};
  }

  template <typename T>
  rajaperf::AutoDataMover<T> scopedMoveData(T*& ptr, Size_type len, VariantID vid)
  {
    DataSpace ds = getDataSpace(vid);
    DataSpace hds = rajaperf::hostCopyDataSpace(ds);
    rajaperf::moveData(hds, ds, ptr, len, getDataAlignment());
    return {ds, hds, ptr, len, getDataAlignment()};
  }

  template <typename T>
  void deallocData(T*& ptr, VariantID vid)
  {
    rajaperf::deallocData(getDataSpace(vid), ptr);
  }

  template <typename T>
  void initData(T& d, VariantID vid)
  {
    (void)vid;
    rajaperf::detail::initData(d);
  }

  template <typename T>
  void addToChecksum(T val)
  {
    checksum += static_cast<Checksum_type>(std::abs(val));
  }

  template <typename T>
  void addToChecksum(T* ptr, Size_type len, VariantID vid)
  {
    addToChecksum(getDataSpace(vid), ptr, len);
  }

  template <typename T>
  void addToChecksum(DataSpace dataSpace, T* ptr, Size_type len)
  {
    checksum += rajaperf::calcChecksum(dataSpace, ptr, len, getDataAlignment());
  }

  void startTimer()
  {
    synchronize();
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timer.start();
    CALI_START;
  }

  void stopTimer()
  {
    synchronize();
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    CALI_STOP; timer.stop(); recordExecTime();
  }

  void resetTimer() { timer.reset(); }

  void print(std::ostream& os) const;

  void runKernel(VariantID vid, size_t tune_idx);

  //
  // Virtual and pure virtual methods that may/must be implemented
  // by concrete kernel subclass.
  //

  virtual void setSize(Index_type target_size, Index_type target_reps) = 0;
  virtual void setUp(VariantID vid, size_t tune_idx) = 0;
  virtual void updateChecksum(VariantID vid, size_t tune_idx) = 0;
  virtual void tearDown(VariantID vid, size_t tune_idx) = 0;

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  void caliperOn() { doCaliperTiming = true; }
  void caliperOff() { doCaliperTiming = false; }
  bool isCaliperTiming() const { return doCaliperTiming; }
  void doOnceCaliMetaBegin(VariantID vid, size_t tune_idx);
  void doOnceCaliMetaEnd(VariantID vid, size_t tune_idx);
  static void setCaliperMgrVariantTuning(VariantID vid,
                                    std::string tstr,
                                    const std::string& outfile,
                                    const std::string& addToSpotConfig,
                                    const std::string& addToCaliConfig,
                                    const int num_variants_tunings);

  static void setCaliperMgrStart(VariantID vid, std::string tstr) { mgr[vid][tstr].start(); }
  static void setCaliperMgrStop(VariantID vid, std::string tstr) { mgr[vid][tstr].stop(); }
  static void setCaliperMgrFlush()
  { // we're going to flush all the variants at once
    for(auto const &mp : mgr) {
      for(auto const &kv : mp.second) {
        // set Adiak key first
        adiak::catstring variant = getVariantName(mp.first);
        adiak::catstring tstr = kv.first;
        adiak::value("variant", variant);
        adiak::value("tuning", tstr);
        mgr[mp.first][kv.first].flush();
      }
    }
  }

  std::string getKernelGroupName(const std::string &kname )
  {
    std::size_t found = kname.find("_");
    return kname.substr(0,found);
  }

#endif

protected:
  const RunParams& run_params;

#if defined(RAJA_PERFSUITE_USE_CALIPER) && \
    defined(RAJA_PERFSUITE_USE_CALIPER_SUBKERNEL)
  cali_id_t Subkernel_attr;
#endif

  struct ChecksumTolerance
  {
    static constexpr inline Checksum_type zero = 0.0;
    static constexpr inline Checksum_type tight = 1e-14;
    static constexpr inline Checksum_type normal = 1e-10;
    static constexpr inline Checksum_type loose = 5e-6;
  };

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  int did;
  int hid;
#endif

private:
  using variant_tuning_method_pointer = void(KernelBase::*)(VariantID);

  KernelBase() = delete;

  void recordExecTime();

  // This is used to implement tunings by being a wrapper for calling
  // the given derived class method by casting this to the derived class.
  // Instantiations of this method are stored in variant_tuning_methods.
  template < typename Derived, void (Derived::* method)(VariantID) >
  void wrapDerivedVariantTuningMethod(VariantID vid)
  {
    Derived& self = dynamic_cast<Derived&>(*this);

    (self.*method)(vid);
  }

  void addVariantTuning(VariantID vid, std::string name, TuningAttribute attrs,
                        variant_tuning_method_pointer method);

  //
  // Boolean member shared by all kernel objects indicating whether they
  // will be run for warmup purposes (true) or not (false).
  //
  static inline bool s_warmup_run = false;

  //
  // Persistent properties of kernel, independent of run
  //
  KernelID    kernel_id = NumKernels;
  std::string name = "unnamed";

  Index_type default_prob_size = -1;
  Index_type default_reps = -1;

  Index_type actual_prob_size = -1;
  Index_type actual_reps = -1;

  bool uses_feature[NumFeatures] = {};

  ChecksumConsistency checksum_consistency = ChecksumConsistency::NumChecksumConsistencies;
  Checksum_type checksum_tolerance = ChecksumTolerance::normal;
  RAJA::KahanSum<Checksum_type> checksum;

  std::vector<Checksum_type> checksum_min[NumVariants];
  std::vector<Checksum_type> checksum_max[NumVariants];
  std::vector<RAJA::KahanSum<Checksum_type>> checksum_sum[NumVariants];

  Complexity complexity = Complexity::NumComplexities;

  Index_type num_nested_perfect_loops = -1;
  Index_type problem_dimensionality = -1;

  std::vector<std::string> variant_tuning_names[NumVariants];
  std::vector<TuningAttribute> variant_tuning_attrs[NumVariants];
  std::vector<variant_tuning_method_pointer> variant_tuning_methods[NumVariants];

  //
  // Properties of kernel dependent on how kernel is run
  //
  Index_type its_per_rep = -1;
  Index_type kernels_per_rep = -1;
  Index_type bytes_allocated_per_rep = -1;
  Index_type bytes_read_per_rep = -1;
  Index_type bytes_written_per_rep = -1;
  Index_type bytes_modify_written_per_rep = -1;
  Index_type bytes_atomic_modify_written_per_rep = -1;
  Index_type FLOPs_per_rep = -1;
  double kernel_block_size = nan(""); // Set default value for non GPU kernels

  VariantID running_variant = NumVariants;
  size_t running_tuning = getUnknownTuningIdx();

  Checksum_type checksum_reference = nan("");
  VariantID checksum_reference_variant = NumVariants;
  size_t checksum_reference_tuning = getUnknownTuningIdx();
  TuningAttribute checksum_reference_tuning_attributes = TuningAttribute::none;

  std::vector<int> num_exec[NumVariants];

  RAJA::Timer timer;

#if defined(RAJA_PERFSUITE_USE_CALIPER)
  bool doCaliperTiming = true; // warmup can use this to exclude timing
  std::vector<bool> doCaliMetaOnce[NumVariants];
  cali_id_t ProblemSize_attr; // in ctor cali_create_attribute("ProblemSize",CALI_TYPE_DOUBLE,CALI_ATTR_ASVALUE | CALI_ATTR_AGGREGATABLE | CALI_ATTR_SKIP_EVENTS);
  cali_id_t Reps_attr;
  cali_id_t Iters_Rep_attr;
  cali_id_t Kernels_Rep_attr;
  cali_id_t Bytes_Allocated_Rep_attr;
  cali_id_t Bytes_Moved_Rep_attr;
  cali_id_t Bytes_Touched_Rep_attr;
  cali_id_t Bytes_Read_Rep_attr;
  cali_id_t Bytes_Written_Rep_attr;
  cali_id_t Bytes_ModifyWritten_Rep_attr;
  cali_id_t Bytes_AtomicModifyWritten_Rep_attr;
  cali_id_t Flops_Rep_attr;
  cali_id_t BlockSize_attr;
  std::map<std::string, cali_id_t> Feature_attrs;
  cali_id_t ChecksumConsistency_attr;
  cali_id_t Complexity_attr;
  cali_id_t MaxPerfectLoopDimensions_attr;
  cali_id_t ProblemDimensionality_attr;


  // we need a Caliper Manager object per variant
  // we can inline this with c++17
  static std::map<rajaperf::VariantID, std::map<std::string, cali::ConfigManager>> mgr;
#endif

  std::vector<RAJA::Timer::ElapsedType> min_time[NumVariants];
  std::vector<RAJA::Timer::ElapsedType> max_time[NumVariants];
  std::vector<RAJA::Timer::ElapsedType> tot_time[NumVariants];
};


// Define the define*VariantTunings function with the given variants for
// the default tuning.
//
// KERNEL is the name of the kernel type (e.g. DAXPY)
// VariantName is the name of the Variant (e.g. Seq)
// ... the names of the variants to add variant tunings for (e.g. Base_Seq, Lambda_Seq, RAJA_Seq)
//
// Example:
// RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DAXPY, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)
//
#define RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(KERNEL, VariantName, ...)   \
  void KERNEL::define##VariantName##VariantTunings()                           \
  {                                                                            \
    for (VariantID vid : {__VA_ARGS__}) {                                      \
      addVariantTuning<&KERNEL::run##VariantName##Variant>(                    \
          vid, getDefaultTuningName());                                        \
    }                                                                          \
  }


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
