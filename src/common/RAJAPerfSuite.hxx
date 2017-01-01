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
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerfSuite_HXX
#define RAJAPerfSuite_HXX

#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
#endif

namespace rajaperf
{

class KernelBase;


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each sub-suite in full suite.
 *
 * IMPORTANT: This is only modified when a sub-suite is added or removed.
 *
 *            ENUM VALUES MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) 
 *            WITH ARRAY OF SUB_SUITE NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum SuiteID {

  Basic = 0,
  Livloops,
  Polybench,
  Stream,
  Apps,

  NumSuites // Keep this one last and DO NOT remove (!!)

};


//
/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each KERNEL in full suite.
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
#if 0
  Basic_INIT3 = 0,
#endif
  Basic_MULADDSUB = 0,
#if 0
  Basic_IF_QUAD,
  Basic_TRAP_INT,
#endif

//
// Livloops kernels...
//
#if 0
  Livloops_HYDRO_1D,
  Livloops_ICCG,
  Livloops_INNER_PROD,
  Livloops_BAND_LIN_EQ,
  Livloops_TRIDIAG_ELIM,
  Livloops_EOS,
  Livloops_ADI,
  Livloops_INT_PREDICT,
  Livloops_DIFF_PREDICT,
  Livloops_FIRST_SUM,
  Livloops_FIRST_DIFF,
  Livloops_PIC_2D,
  Livloops_PIC_1D,
  Livloops_HYDRO_2D,
  Livloops_GEN_LIN_RECUR,
  Livloops_DISC_ORD,
  Livloops_MAT_X_MAT,
  Livloops_PLANCKIAN,
  Livloops_IMP_HYDRO_2D,
  Livloops_FIND_FIRST_MIN,
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
#if 0
  Stream_***
#endif

//
// Apps kernels...
//
#if 0
  Apps_PRESSURE_CALC,
  Apps_ENERGY_CALC,
  Apps_VOL3D_CALC,
  Apps_DEL_DOT_VEC_2D,
  Apps_COUPLE,
  Apps_FIR,
#endif

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

  Baseline = 0,
  RAJA_Serial,
  Baseline_OpenMP,
  RAJA_OpenMP,
  Baseline_CUDA,
  RAJA_CUDA,

  NumVariants // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Return suite name associated with SuiteID enum value.
 *
 *******************************************************************************
 */
const std::string& getSuiteName(SuiteID sid);

/*!
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
const std::string& getKernelName(KernelID kid);

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
KernelBase* getKernelObject(KernelID kid,
                            double sample_frac,
                            double size_frac);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
