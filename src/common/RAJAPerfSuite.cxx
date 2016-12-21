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
  std::string("basic_INIT3"),
#endif
  std::string("basic_MULADDSUB"),
#if 0
  std::string("basic_IF_QUAD"),
  std::string("basic_TRAP_INT"),
#endif

//
// Livloops kernels...
//
#if 0
  std::string("livloops_HYDRO_1D"),
  std::string("livloops_ICCG"),
  std::string("livloops_INNER_PROD"),
  std::string("livloops_BAND_LIN_EQ"),
  std::string("livloops_TRIDIAG_ELIM"),
  std::string("livloops_EOS"),
  std::string("livloops_ADI"),
  std::string("livloops_INT_PREDICT"),
  std::string("livloops_DIFF_PREDICT"),
  std::string("livloops_FIRST_SUM"),
  std::string("livloops_FIRST_DIFF"),
  std::string("livloops_PIC_2D"),
  std::string("livloops_PIC_1D"),
  std::string("livloops_HYDRO_2D"),
  std::string("livloops_GEN_LIN_RECUR"),
  std::string("livloops_DISC_ORD"),
  std::string("livloops_MAT_X_MAT"),
  std::string("livloops_PLANCKIAN"),
  std::string("livloops_IMP_HYDRO_2D"),
  std::string("livloops_FIND_FIRST_MIN"),
#endif

//
// Polybench kernels...
//
#if 0
  std::string("polybench_***");
#endif

//
// Stream kernels...
//
#if 0
  std::string("stream_***");
#endif

//
// Apps kernels...
//
#if 0
  std::string("apps_PRESSURE_CALC"),
  std::string("apps_ENERGY_CALC"),
  std::string("apps_VOL3D_CALC"),
  std::string("apps_DEL_DOT_VEC_2D"),
  std::string("apps_COUPLE"),
  std::string("apps_FIR"),
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
  std::string("RAJA_SEQ"),
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
    case basic_INIT3 : {
       kernel = new basic::INIT3();
       break;
    }
#endif
    case basic_MULADDSUB : {
       kernel = new basic::MULADDSUB(sample_frac, size_frac);
       break;
    }
#if 0
    case basic_IF_QUAD : {
       kernel = new basic::IF_QUAD();
       break;
    }
    case basic_TRAP_INT : {
       kernel = new basic::TRAP_INT();
       break;
    }
#endif

//
// Livloops kernels...
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

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
