//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Types and methods for managing Suite kernels, variants, features, etc..
///

#ifndef RAJAPerfSuite_HPP
#define RAJAPerfSuite_HPP

#include "RAJA/config.hpp"
#include "rajaperf_config.hpp"

#include <string>
#include <ostream>

namespace rajaperf
{

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
  Basic_DAXPY_ATOMIC,
  Basic_IF_QUAD,
  Basic_INDEXLIST,
  Basic_INDEXLIST_3LOOP,
  Basic_INIT3,
  Basic_INIT_VIEW1D,
  Basic_INIT_VIEW1D_OFFSET,
  Basic_MAT_MAT_SHARED,
  Basic_MULADDSUB,
  Basic_NESTED_INIT,
  Basic_PI_ATOMIC,
  Basic_PI_REDUCE,
  Basic_REDUCE3_INT,
  Basic_REDUCE_STRUCT,
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
//
  Polybench_2MM,
  Polybench_3MM,
  Polybench_ADI,
  Polybench_ATAX,
  Polybench_FDTD_2D,
  Polybench_FLOYD_WARSHALL,
  Polybench_GEMM,
  Polybench_GEMVER,
  Polybench_GESUMMV,
  Polybench_HEAT_3D,
  Polybench_JACOBI_1D,
  Polybench_JACOBI_2D,
  Polybench_MVT,

//
// Stream kernels...
//
  Stream_ADD,
  Stream_COPY,
  Stream_DOT,
  Stream_MUL,
  Stream_TRIAD,

//
// Apps kernels...
//
  Apps_CONVECTION3DPA,
  Apps_COUPLE,
  Apps_DEL_DOT_VEC_2D,
  Apps_DIFFUSION3DPA,
  Apps_ENERGY,
  Apps_FIR,
  Apps_HALOEXCHANGE,
  Apps_HALOEXCHANGE_FUSED,
  Apps_LTIMES,
  Apps_LTIMES_NOVIEW,
  Apps_MASS3DPA,
  Apps_NODAL_ACCUMULATION_3D,
  Apps_PRESSURE,
  Apps_VOL3D,

//
// Algorithm kernels...
//
  Algorithm_SCAN,
  Algorithm_SORT,
  Algorithm_SORTPAIRS,
  Algorithm_REDUCE_SUM,
  Algorithm_MEMSET,
  Algorithm_MEMCPY,

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

  Base_StdPar,
  Lambda_StdPar,
  RAJA_StdPar,

  Kokkos_Lambda,

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
 * \brief Return true if variant associated with VariantID enum value runs
 *        on the gpu.
 *
 *******************************************************************************
 */
bool isVariantGPU(VariantID vid);

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

/*!
 *******************************************************************************
 *
 * \brief Return ostream used as cout.
 *
 *        IMPORTANT: May return a non-printing stream when MPI is enabled.
 *
 *******************************************************************************
 */
std::ostream& getCout();

/*!
 *******************************************************************************
 *
 * \brief Return non-printing ostream.
 *
 *******************************************************************************
 */
std::ostream* makeNullStream();

/*!
 *******************************************************************************
 *
 * \brief Return reference to global non-printing ostream.
 *
 *******************************************************************************
 */
std::ostream& getNullStream();

/*!
 *******************************************************************************
 *
 * \brief Empty function used to squash compiler warnings for unused variables.
 *
 *******************************************************************************
 */
template < typename... Ts >
inline void ignore_unused(Ts&&...) { }

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
