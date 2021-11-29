//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Declare types, methods and namespaces to enable RAJAPerf Suite to handle Kokkos kernels, variants, features, etc.

#ifndef RAJAPerfSuite_HPP
#define RAJAPerfSuite_HPP

#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "RAJA/config.hpp"

#if defined(RUN_KOKKOS)
#include "Kokkos_Core.hpp"
#endif // RUN_KOKKOS
#endif // RAJAPERF_INFRASTRUCTURE_ONLY

#include <string>
namespace rajaperf
{
class RunParams;
class Executor;
class KernelBase;

const RunParams& getRunParams(Executor*);
void free_register_group(Executor*, std::string);
void free_register_kernel(Executor*, std::string, KernelBase*);
void make_perfsuite_executor(Executor* exec, int argc, char* argv[]);

#if defined(RUN_KOKKOS)
template <class PointedAt, size_t NumBoundaries>
struct PointerOfNdimensions;

template <class PointedAt>
struct PointerOfNdimensions<PointedAt, 0> {
  using type = PointedAt;
};

template <class PointedAt, size_t NumBoundaries>
struct PointerOfNdimensions {
  using type =
      typename PointerOfNdimensions<PointedAt, NumBoundaries - 1>::type *;
};

// This templated function is used to wrap pointers (declared and defined in RAJAPerf Suite kernels) in Kokkos Views
template <class PointedAt, class... Boundaries>
auto getViewFromPointer(PointedAt *kokkos_ptr, Boundaries... boundaries)
    -> typename Kokkos::View<
       typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
       typename Kokkos::DefaultExecutionSpace::memory_space>

{

  using host_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  // Nota bene: When copying data, we can either change the Layout or the memory_space
  // (host or device), but we cannot change both!
  // Here, we are mirroring data on the (CPU) host TO the (GPU) device, i.e., Layout is
  // as if on the device, but the data actually reside on the host.  The host
  // mirror will be Layout Left (optimal for the device, but not the host).

  using mirror_view_type = typename device_view_type::HostMirror;

  // Assignment statement: we are constructing a host_view_type called
  // pointer_holder.  The value of kokkos_ptr is the Kokkos View-wrapped pointer
  // on the Host (CPU), and the Boundaries parameter pack values, boundaries (i.e., array boundaries) will also
  // be part of this this host_view_type object.

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // The boundaries parameter pack contains the array dimenions; 
  // an allocation is implicitly made here
  device_view_type device_data_copy("StringName", boundaries...);

  mirror_view_type cpu_to_gpu_mirror =
      Kokkos::create_mirror_view(device_data_copy);

  // deep_copy our existing data, the contents of
  // pointer_holder, into the mirror_view;
  // Copying from Host to Device has two steps:
  // 	1) Change the layout to enable sending data from CPU to GPU
  // 	2) Change the memory_space (host or device) to send the optimal data
  // 	layout to the GPU.  
  
  // This step changes the array layout to be optimal for the gpu, i.e.,
  // LayoutLeft.
  Kokkos::deep_copy(cpu_to_gpu_mirror, pointer_holder);

  // The mirror view data layout on the HOST is like the layout for the GPU.
  // GPU-optimized layouts are LayoutLeft, i.e., column-major This deep_copy
  // copy GPU-layout data on the HOST to the Device

  // Actual copying of the data from the host to the gpu
  Kokkos::deep_copy(device_data_copy, cpu_to_gpu_mirror);

  // Kokkos::View return type

  return device_data_copy;
}

// This function will move data in a Kokkos::View back to host from device,
// and will store in the existing pointer(s)
template <class PointedAt, class ExistingView, class... Boundaries>
void moveDataToHostFromKokkosView(PointedAt *kokkos_ptr, ExistingView my_view,
                                  Boundaries... boundaries)
{

  using host_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  using mirror_view_type = typename device_view_type::HostMirror;

  // Constructing a host_view_type with the name
  // pointer_holder.  The contents/value of kokkos_ptr is the pointer we're wrapping on
  // the Host, and the Boundaries parameter pack values, boundaries, will also
  // be part of this this host_view_type object.

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // Layout is optimal for gpu, but data are actually located on CPU
  mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(my_view);

  // Actual copying of the data from the gpu (my_view) back to the cpu
  Kokkos::deep_copy(cpu_to_gpu_mirror, my_view);

  // This copies from the mirror on the host cpu back to the existing
  // pointer(s)
  Kokkos::deep_copy(pointer_holder, cpu_to_gpu_mirror);
}

#endif // RUN_KOKKOS

class KernelBase;
class RunParams;


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each group of kernels in suite.
 *
 * IMPORTANT: This is only modified when a group is added or removed.
 *
 *            ENUM VALUES MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) 
 *            WITH ARRAY OF GROUP NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum GroupID {

  Basic = 0,
  Lcals,
  Polybench,
  Stream,
  Apps,
  Algorithm,

  NumGroups // Keep this one last and DO NOT remove (!!)

};


//
/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a kernel is added or removed.
 *
 *            ENUM VALUES MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) 
 *            WITH ARRAY OF KERNEL NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum KernelID {

//
// Basic kernels...
//
  Basic_DAXPY = 0,
  Basic_IF_QUAD,
  Basic_INIT3,
  Basic_INIT_VIEW1D,
  Basic_INIT_VIEW1D_OFFSET,
  Basic_MAT_MAT_SHARED,
  Basic_MULADDSUB,
  Basic_NESTED_INIT,
  Basic_PI_ATOMIC,
  Basic_PI_REDUCE,
  Basic_REDUCE3_INT,
  Basic_TRAP_INT,

//
// Lcals kernels...
//
  Lcals_DIFF_PREDICT,
  Lcals_EOS,
  Lcals_FIRST_DIFF,
  Lcals_FIRST_MIN,
  Lcals_FIRST_SUM,
  Lcals_GEN_LIN_RECUR,
  Lcals_HYDRO_1D,
  Lcals_HYDRO_2D,
  Lcals_INT_PREDICT,
  Lcals_PLANCKIAN,
  Lcals_TRIDIAG_ELIM,

//
// Polybench kernels...
// These will be uncommented once Kokkos translations for these kernels exist
//
//  Polybench_2MM,
//  Polybench_3MM,
//  Polybench_ADI,
//  Polybench_ATAX,
//  Polybench_FDTD_2D,
//  Polybench_FLOYD_WARSHALL,
//  Polybench_GEMM,
//  Polybench_GEMVER,
//  Polybench_GESUMMV,
//  Polybench_HEAT_3D,
//  Polybench_JACOBI_1D,
//  Polybench_JACOBI_2D,
//  Polybench_MVT,


// Stream kernels...
//
  Stream_ADD,
  Stream_COPY,
  Stream_DOT,
  Stream_MUL,
  Stream_TRIAD,

//
// Apps kernels...
  //Apps_COUPLE,
  Apps_DEL_DOT_VEC_2D,
  Apps_DIFFUSION3DPA,
  Apps_ENERGY,
  Apps_FIR,
  Apps_HALOEXCHANGE,
  Apps_HALOEXCHANGE_FUSED,
  Apps_LTIMES,
  Apps_LTIMES_NOVIEW,
  Apps_MASS3DPA,
  Apps_PRESSURE,
  Apps_VOL3D,

//
// Algorithm kernels...
//
  Algorithm_SORT,
  Algorithm_SORTPAIRS,

  NumKernels // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new variant is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ARRAY OF VARIANT NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum VariantID {

  Base_Seq = 0,
  Lambda_Seq,
  RAJA_Seq,

  Base_OpenMP,
  Lambda_OpenMP,
  RAJA_OpenMP,

  Base_OpenMPTarget,
  RAJA_OpenMPTarget,

  Base_CUDA,
  Lambda_CUDA,
  RAJA_CUDA,

  Base_HIP,
  Lambda_HIP,
  RAJA_HIP,

  Kokkos_Lambda,
  Kokkos_Functor,

  NumVariants // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each (RAJA) FEATURE used in suite.
 *
 * IMPORTANT: This is only modified when a new feature is used in suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ARRAY OF FEATURE NAMES IN IMPLEMENTATION FILE!!!
 *
 *******************************************************************************
 */
enum FeatureID {

  Forall = 0,
  Kernel,
  Teams,

  Sort,
  Scan,
  Workgroup, 

  Reduction,
  Atomic,

  View,

  NumFeatures // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Return group name associated with GroupID enum value.
 *
 *******************************************************************************
 */
const std::string& getGroupName(GroupID gid);

/*!
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 * Kernel name is full kernel name (see below) with group name prefix removed.
 *
 *******************************************************************************
 */
std::string getKernelName(KernelID kid);

/*!
 *******************************************************************************
 *
 * \brief Return full kernel name associated with KernelID enum value.
 *
 * Full kernel name is <group name>_<kernel name>.
 *
 *******************************************************************************
 */
const std::string& getFullKernelName(KernelID kid);

/*!
 *******************************************************************************
 *
 * \brief Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
const std::string& getVariantName(VariantID vid); 

/*!
 *******************************************************************************
 *
 * \brief Return true if variant associated with VariantID enum value is 
 *        available * to run; else false.
 *
 *******************************************************************************
 */
bool isVariantAvailable(VariantID vid);

/*!
 *******************************************************************************
 *
 * \brief Return feature name associated with FeatureID enum value.
 *
 *******************************************************************************
 */
const std::string& getFeatureName(FeatureID vid);

/*!
 *******************************************************************************
 *
 * \brief Construct and return kernel object for given KernelID enum value.
 *
 *        IMPORTANT: Caller assumes ownerhip of returned object.
 *
 *******************************************************************************
 */
KernelBase* getKernelObject(KernelID kid, const RunParams& run_params);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
