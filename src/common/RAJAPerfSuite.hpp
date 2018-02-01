//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Tyoes and methods for managing Suite kernels and variants.
///

#ifndef RAJAPerfSuite_HPP
#define RAJAPerfSuite_HPP

#include "RAJA/config.hpp"

#include <string>

namespace rajaperf
{

class KernelBase;
class RunParams;

/*!
 *******************************************************************************
 *
 * \brief Enumeration defining size specification for the polybench kernels
 *
 * Polybench comes with a spec file to setup the iteration space for the various sizes:
 * Mini,Small,Medium,Large,Extralarge
 *
 * and we adapt those entries within this perfsuite.
 *
 * The default size is Medium, but can be overriding as a parameter at run-time
 *
 * An example partial entry from that file showing the MINI and SMALL spec for the
 * kernel 3mm
 *
 * kernel	category	datatype	params	MINI	SMALL	MEDIUM	LARGE	EXTRALARGE
 * 3mm	linear-algebra/kernels	double	NI NJ NK NL NM	16 18 20 22 24	40 50 60 70 80 .... 
 * *
 *******************************************************************************
 */
typedef enum SizeSpec {Mini,Small,Medium,Large,Extralarge,Specundefined} SizeSpec_T;


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
  Basic_MULADDSUB = 0,
  Basic_IF_QUAD,
  Basic_TRAP_INT,
  Basic_INIT3,
  Basic_REDUCE3_INT,
  Basic_NESTED_INIT,
  Basic_INIT_VIEW1D,
  Basic_INIT_VIEW1D_OFFSET,

//
// Lcals kernels...
//
  Lcals_HYDRO_1D,
  Lcals_EOS,
  Lcals_INT_PREDICT,
  Lcals_DIFF_PREDICT,
  Lcals_FIRST_DIFF,
  Lcals_PLANCKIAN,

//
// Polybench kernels...
//
  Polybench_2MM,
  Polybench_3MM,
  Polybench_GEMMVER,

//
// Stream kernels...
//
  Stream_COPY,
  Stream_MUL,
  Stream_ADD,
  Stream_TRIAD,
  Stream_DOT,

//
// Apps kernels...
//
  Apps_PRESSURE,
  Apps_ENERGY,
  Apps_VOL3D,
  Apps_DEL_DOT_VEC_2D,
  Apps_FIR,
  Apps_LTIMES,
  Apps_LTIMES_NOVIEW,
  Apps_COUPLE,

  NumKernels // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ARRAY OF VARIANT NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum VariantID {

  Base_Seq = 0,
  RAJA_Seq,

#if defined(RAJA_ENABLE_OPENMP)
  Base_OpenMP,
  RAJA_OpenMP,

#if defined(RAJA_ENABLE_TARGET_OPENMP)  
  Base_OpenMPTarget,
  RAJA_OpenMPTarget,
#endif

#endif

#if defined(RAJA_ENABLE_CUDA)
  Base_CUDA,
  RAJA_CUDA,
#endif

  NumVariants // Keep this one last and NEVER comment out (!!)

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
 * \brief Construct and return kernel object for given KernelID enum value.
 *
 *        IMPORTANT: Caller assumes ownerhip of returned object.
 *
 *******************************************************************************
 */
KernelBase* getKernelObject(KernelID kid, const RunParams& run_params);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
