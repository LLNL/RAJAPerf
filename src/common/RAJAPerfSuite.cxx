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
  std::string("Basic_INIT3"),
#endif
  std::string("Basic_MULADDSUB"),
#if 0
  std::string("Basic_IF_QUAD"),
  std::string("Basic_TRAP_INT"),
#endif

//
// Lloops kernels...
//
#if 0
  std::string("LLoops_HYDRO_1D"),
  std::string("LLoops_ICCG"),
  std::string("LLoops_INNER_PROD"),
  std::string("LLoops_BAND_LIN_EQ"),
  std::string("LLoops_TRIDIAG_ELIM"),
  std::string("LLoops_EOS"),
  std::string("LLoops_ADI"),
  std::string("LLoops_INT_PREDICT"),
  std::string("LLoops_DIFF_PREDICT"),
  std::string("LLoops_FIRST_SUM"),
  std::string("LLoops_FIRST_DIFF"),
  std::string("LLoops_PIC_2D"),
  std::string("LLoops_PIC_1D"),
  std::string("LLoops_HYDRO_2D"),
  std::string("LLoops_GEN_LIN_RECUR"),
  std::string("LLoops_DISC_ORD"),
  std::string("LLoops_MAT_X_MAT"),
  std::string("LLoops_PLANCKIAN"),
  std::string("LLoops_IMP_HYDRO_2D"),
  std::string("LLoops_FIND_FIRST_MIN"),
#endif

//
// Polybench kernels...
//
#if 0
  std::string("Polybench_***");
#endif

//
// Stream kernels...
//
#if 0
  std::string("Stream_***");
#endif

//
// Apps kernels...
//
#if 0
  std::string("Apps_PRESSURE_CALC"),
  std::string("Apps_ENERGY_CALC"),
  std::string("Apps_VOL3D_CALC"),
  std::string("Apps_DEL_DOT_VEC_2D"),
  std::string("Apps_COUPLE"),
  std::string("Apps_FIR"),
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
    case Basic_INIT3 : {
       kernel = new basic::INIT3();
       break;
    }
#endif
    case Basic_MULADDSUB : {
       kernel = new basic::MULADDSUB();
       break;
    }
#if 0
    case Basic_IF_QUAD : {
       kernel = new basic::IF_QUAD();
       break;
    }
    case Basic_TRAP_INT : {
       kernel = new basic::TRAP_INT();
       break;
    }
#endif

//
// Lloops kernels...
//
#if 0
  LLoops_HYDRO_1D,
  LLoops_ICCG,
  LLoops_INNER_PROD,
  LLoops_BAND_LIN_EQ,
  LLoops_TRIDIAG_ELIM,
  LLoops_EOS,
  LLoops_ADI,
  LLoops_INT_PREDICT,
  LLoops_DIFF_PREDICT,
  LLoops_FIRST_SUM,
  LLoops_FIRST_DIFF,
  LLoops_PIC_2D,
  LLoops_PIC_1D,
  LLoops_HYDRO_2D,
  LLoops_GEN_LIN_RECUR,
  LLoops_DISC_ORD,
  LLoops_MAT_X_MAT,
  LLoops_PLANCKIAN,
  LLoops_IMP_HYDRO_2D,
  LLoops_FIND_FIRST_MIN,
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

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
