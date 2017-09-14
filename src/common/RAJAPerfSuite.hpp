/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing enums, names, and interfaces for defining 
 *          performance suite kernels and variants.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


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

//
// Lcals kernels...
//
  Lcals_HYDRO_1D,
  Lcals_EOS,
  Lcals_FIRST_DIFF,
  Lcals_PLANCKIAN,
#if 0
  Lcals_GEN_LIN_RECUR,
#endif

//
// Polybench kernels...
//
#if 0
  Polybench_***
#endif

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
  Apps_COUPLE,
  Apps_FIR,

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
#if defined(ENABLE_OPENMP)
  Base_OpenMP,
  RAJALike_OpenMP,
  RAJA_OpenMP,
#endif
#if defined(ENABLE_CUDA)
  Base_CUDA,
  RAJA_CUDA,
#endif
#if 0
  Base_OpenMP4x,
  RAJA_OpenMP4x,
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
