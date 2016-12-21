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
  basic_INIT3 = 0,
#endif
  basic_MULADDSUB = 0,
#if 0
  basic_IF_QUAD,
  basic_TRAP_INT,
#endif

//
// livloops kernels...
//
#if 0
  livloops_HYDRO_1D,
  livloops_ICCG,
  livloops_INNER_PROD,
  livloops_BAND_LIN_EQ,
  livloops_TRIDIAG_ELIM,
  livloops_EOS,
  livloops_ADI,
  livloops_INT_PREDICT,
  livloops_DIFF_PREDICT,
  livloops_FIRST_SUM,
  livloops_FIRST_DIFF,
  livloops_PIC_2D,
  livloops_PIC_1D,
  livloops_HYDRO_2D,
  livloops_GEN_LIN_RECUR,
  livloops_DISC_ORD,
  livloops_MAT_X_MAT,
  livloops_PLANCKIAN,
  livloops_IMP_HYDRO_2D,
  livloops_FIND_FIRST_MIN,
#endif

//
// Polybench kernels...
//
#if 0
  polybench_***
#endif

//
// Stream kernels...
//
#if 0
  stream_***
#endif

//
// Apps kernels...
//
#if 0
  apps_PRESSURE_CALC,
  apps_ENERGY_CALC,
  apps_VOL3D_CALC,
  apps_DEL_DOT_VEC_2D,
  apps_COUPLE,
  apps_FIR,
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
KernelBase* getKernelObject(KernelID kid,
                            double sample_frac,
                            double size_frac);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
