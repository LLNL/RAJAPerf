/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing data and interfaces for defining 
 *          performance suite kernels, variants, and run parameters.
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

/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *******************************************************************************
 */
enum KernelID {

//
// LCALS kernels...
//
#if 0
   PRESSURE_CALC = 0,
   ENERGY_CALC,
   VOL3D_CALC,
   DEL_DOT_VEC_2D,
   COUPLE,
   FIR,

   INIT3,
#endif
   MULADDSUB = 0,
#if 0
   IF_QUAD,
   TRAP_INT,

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

   NUM_KERNELS // Keep this one last and NEVER comment out (!!)

};


/*!
 *******************************************************************************
 *
 * \brief Enumeration defining unique id for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
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
 * \brief Simple class to hold suite execution parameters.
 *
 *******************************************************************************
 */
class RunParams {
public:
  RunParams( int argc, char** argv );
  ~RunParams( );

  int npasses;                     /*!< Number of passes through suite.  */

  std::string run_kernels;         /*!< Filter which kernels to run... */
  std::string run_variants;        /*!< Filter which variants to run... */

  double length_fraction;          /*!< Fraction of default kernel length run */

  std::string output_file_prefix;  /*!< Prefix for output data file. */

#if 0 // RDH TODO
  std::vector<KernelBase*> kernels;/*!< Vector of kernel objects to run */
#endif
  std::vector<VariantID> variants; /*!< Vector of variant IDs to run */

private:
// These are not implemented to prevent calling by accident.
  RunParams();

};


/*!
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
std::string getKernelName(KernelID kid);

#if 0 // RDH TODO
/*!
 *******************************************************************************
 *
 * \brief Return kernel object associated with KernelID enum value.
 *
 *******************************************************************************
 */
KernelBase getKernelObject(KernelID kid);
#endif

/*!
 *******************************************************************************
 *
 * \brief Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
std::string getVariantName(VariantID vid);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
