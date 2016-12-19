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

// Std C++ includes
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
#endif

#ifndef RAJAPerfSuite_HXX
#define RAJAPerfSuite_HXX

namespace rajaperf
{

class KernelBase;

/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ARRAY OF KERNEL NAMES IN IMPLEMENTATION FILE!!! 
 *
 *******************************************************************************
 */
enum KernelID {

//
// Basic kernels...
//
#if 0
  INIT3 = 0,
#endif
  MULADDSUB = 0,
#if 0
  IF_QUAD,
  TRAP_INT,
#endif

//
// Lloops kernels...
//
#if 0
  HYDRO_1D,
  ICCG,
  INNER_PROD,
  BAND_LIN_EQ,
  TRIDIAG_ELIM,
  EOS,
  ADI,
  INT_PREDICT,
  DIFF_PREDICT,
  FIRST_SUM,
  FIRST_DIFF,
  PIC_2D,
  PIC_1D,
  HYDRO_2D,
  GEN_LIN_RECUR,
  DISC_ORD,
  MAT_X_MAT,
  PLANCKIAN,
  IMP_HYDRO_2D,
  FIND_FIRST_MIN,
#endif

//
// Polybench kernels...
//

//
// Stream kernels...
//

//
// Apps kernels...
//
#if 0
  PRESSURE_CALC,
  ENERGY_CALC,
  VOL3D_CALC,
  DEL_DOT_VEC_2D,
  COUPLE,
  FIR,
#endif

  NUM_KERNELS // Keep this one last and NEVER comment out (!!)

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

  BASELINE = 0,
  RAJA_SERIAL,
  BASELINE_OPENMP,
  RAJA_OPENMP,
  BASELINE_CUDA,
  RAJA_CUDA,

  NUM_VARIANTS // Keep this one last and NEVER comment out (!!)

};


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
KernelBase* getKernelObject(KernelID kid);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
