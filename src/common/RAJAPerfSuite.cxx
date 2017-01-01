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
 * \brief Array of names for each SUB-SUITE in full suite.
 *
 * IMPORTANT: This is only modified when a sub-suite is added or removed.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF SUITE IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string SuiteNames [] =
{
  std::string("Basic"),
  std::string("Livloops"),
  std::string("Polybench"),
  std::string("Stream"),
  std::string("Apps"),

  std::string("Unknown Kernel")  // Keep this at the end and DO NOT remove....

}; // END SuiteNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each KERNEL in suite.
 *
 * IMPORTANT: This is only modified when a kernel is added or removed.
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
  std::string("Basic_INIT3"),
  std::string("Basic_MULADDSUB"),
  std::string("Basic_IF_QUAD"),
  std::string("Basic_TRAP_INT"),

//
// Livloops kernels...
//
  std::string("Livloops_HYDRO_1D"),
  std::string("Livloops_ICCG"),
  std::string("Livloops_INNER_PROD"),
  std::string("Livloops_BAND_LIN_EQ"),
  std::string("Livloops_TRIDIAG_ELIM"),
  std::string("Livloops_EOS"),
  std::string("Livloops_ADI"),
  std::string("Livloops_INT_PREDICT"),
  std::string("Livloops_DIFF_PREDICT"),
  std::string("Livloops_FIRST_SUM"),
  std::string("Livloops_FIRST_DIFF"),
  std::string("Livloops_PIC_2D"),
  std::string("Livloops_PIC_1D"),
  std::string("Livloops_HYDRO_2D"),
  std::string("Livloops_GEN_LIN_RECUR"),
  std::string("Livloops_DISC_ORD"),
  std::string("Livloops_MAT_X_MAT"),
  std::string("Livloops_PLANCKIAN"),
  std::string("Livloops_IMP_HYDRO_2D"),
  std::string("Livloops_FIND_FIRST_MIN"),

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

  std::string("Unknown Kernel")  // Keep this at the end and DO NOT remove....

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

  std::string("Baseline"),
  std::string("RAJA_Serial"),
  std::string("Baseline_OpenMP"),
  std::string("RAJA_OpenMP"),
  std::string("Baseline_CUDA"),
  std::string("RAJA_CUDA"),

  std::string("Unknown Variant")  // Keep this at the end and DO NOT remove....

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
KernelBase* getKernelObject(KernelID kid,
                            double sample_frac,
                            double size_frac)
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
       kernel = new basic::MULADDSUB(sample_frac, size_frac);
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

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
