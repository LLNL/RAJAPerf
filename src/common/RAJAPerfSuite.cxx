/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file containing names of suite kernels and 
 *          variants, and routine for creating kernel objects.
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


#include "RAJAPerfSuite.hxx"

//
// LCALS kernels...
//
#if 0
#include "basic/INIT3.hxx"
#endif
#include "basic/MULADDSUB.hxx"

#include <iostream>

namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Array of names for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF KERNEL IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string KernelNames [] =
{

//
// Basic kernels...
//
#if 0
  std::string("INIT3"),
#endif
  std::string("MULADDSUB"),
#if 0
  std::string("IF_QUAD"),
  std::string("TRAP_INT"),
#endif

//
// Lloops kernels...
//
#if 0
  std::string("HYDRO_1D"),
  std::string("ICCG"),
  std::string("INNER_PROD"),
  std::string("BAND_LIN_EQ"),
  std::string("TRIDIAG_ELIM"),
  std::string("EOS"),
  std::string("ADI"),
  std::string("INT_PREDICT"),
  std::string("DIFF_PREDICT"),
  std::string("FIRST_SUM"),
  std::string("FIRST_DIFF"),
  std::string("PIC_2D"),
  std::string("PIC_1D"),
  std::string("HYDRO_2D"),
  std::string("GEN_LIN_RECUR"),
  std::string("DISC_ORD"),
  std::string("MAT_X_MAT"),
  std::string("PLANCKIAN"),
  std::string("IMP_HYDRO_2D"),
  std::string("FIND_FIRST_MIN"),
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
  std::string("PRESSURE_CALC"),
  std::string("ENERGY_CALC"),
  std::string("VOL3D_CALC"),
  std::string("DEL_DOT_VEC_2D"),
  std::string("COUPLE"),
  std::string("FIR"),
#endif

  "UNDEFINED"  // Keep this one at the end....

}; // END KernelNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new kernel is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF VARIANT IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string VariantNames [] =
{

  std::string("BASELINE"),
  std::string("RAJA_SEQUENTIAL"),
  std::string("BASELINE_OPENMP"),
  std::string("RAJA_OPENMP"),
  std::string("BASELINE_CUDA"),
  std::string("RAJA_CUDA"),

  "UNDEFINED"  // Keep this one at the end....

}; // END VariantNames


/*
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
const std::string& getKernelName(KernelID kid)
{
  return KernelNames[kid];
}


/*
 *******************************************************************************
 *
 * \brief Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
const std::string& getVariantName(VariantID vid)
{
  return VariantNames[vid];
}

/*
 *******************************************************************************
 *
 * \brief Construct and return kernel object for given KernelID enum value.
 *
 *******************************************************************************
 */
KernelBase* getKernelObject(KernelID kid)
{
  KernelBase* kernel = 0;

  switch ( kid ) {

    //
    // Basic kernels...
    //
#if 0
    case INIT3 : {
       kernel = new basic::INIT3();
       break;
    }
#endif
    case MULADDSUB : {
       kernel = new basic::MULADDSUB();
       break;
    }
#if 0
  IF_QUAD
  TRAP_INT
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

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
