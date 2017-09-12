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
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

//
// Basic kernels...
//
#include "basic/MULADDSUB.hpp"
#include "basic/IF_QUAD.hpp"
#include "basic/TRAP_INT.hpp"
#include "basic/INIT3.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/NESTED_INIT.hpp"

//
// Lcals kernels...
//
#if 0
#include "lcals/HYDRO_1D.hpp"
#endif
#include "lcals/EOS.hpp"
#if 0
#include "lcals/PIC_2D.hpp"
#include "lcals/DISC_ORD.hpp"
#endif

//
// Polybench kernels...
//

//
// Stream kernels...
//
#include "stream/COPY.hpp"
#include "stream/MUL.hpp"
#include "stream/ADD.hpp"
#include "stream/TRIAD.hpp"
#include "stream/DOT.hpp"

//
// Apps kernels...
//
#include "apps/PRESSURE.hpp"
#include "apps/ENERGY.hpp"
#include "apps/VOL3D.hpp"
#include "apps/DEL_DOT_VEC_2D.hpp"
#include "apps/COUPLE.hpp"
#include "apps/FIR.hpp"


#include <iostream>

namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Array of names for each GROUP in suite.
 *
 * IMPORTANT: This is only modified when a group is added or removed.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF GROUP IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string GroupNames [] =
{
  std::string("Basic"),
  std::string("Lcals"),
  std::string("Polybench"),
  std::string("Stream"),
  std::string("Apps"),

  std::string("Unknown Group")  // Keep this at the end and DO NOT remove....

}; // END GroupNames


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
  std::string("Basic_MULADDSUB"),
  std::string("Basic_IF_QUAD"),
  std::string("Basic_TRAP_INT"),
  std::string("Basic_INIT3"),
  std::string("Basic_REDUCE3_INT"),
  std::string("Basic_NESTED_INIT"),

//
// Lcals kernels...
//
#if 0
  std::string("Lcals_HYDRO_1D"),
#endif
  std::string("Lcals_EOS"),
#if 0
  std::string("Lcals_PIC_2D"),
  std::string("Lcals_DISC_ORD"),
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
  std::string("Stream_COPY"),
  std::string("Stream_MUL"),
  std::string("Stream_ADD"),
  std::string("Stream_TRIAD"),
  std::string("Stream_DOT"),

//
// Apps kernels...
//
  std::string("Apps_PRESSURE"),
  std::string("Apps_ENERGY"),
  std::string("Apps_VOL3D"),
  std::string("Apps_DEL_DOT_VEC_2D"),
  std::string("Apps_COUPLE"),
  std::string("Apps_FIR"),

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

  std::string("Base_Seq"),
  std::string("RAJA_Seq"),
#if defined(ENABLE_OPENMP)
  std::string("Base_OpenMP"),
  std::string("RAJALike_OpenMP"),
  std::string("RAJA_OpenMP"),
#endif
#if defined(ENABLE_CUDA)
  std::string("Base_CUDA"),
  std::string("RAJA_CUDA"),
#endif
#if 0
  std::string("Base_OpenMP4.x"),
  std::string("RAJA_OpenMP4.x"),
#endif

  std::string("Unknown Variant")  // Keep this at the end and DO NOT remove....

}; // END VariantNames


/*
 *******************************************************************************
 *
 * \brief Return group name associated with GroupID enum value.
 *
 *******************************************************************************
 */
const std::string& getGroupName(GroupID sid)
{
  return GroupNames[sid];
}


/*
 *******************************************************************************
 *
 * \brief Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
std::string getKernelName(KernelID kid)
{
  std::string::size_type pos = KernelNames[kid].find("_");
  std::string kname(KernelNames[kid].substr(pos+1, std::string::npos));
  return kname;
}


/*
 *******************************************************************************
 *
 * \brief Return full kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
const std::string& getFullKernelName(KernelID kid)
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
                            const RunParams& run_params)
{
  KernelBase* kernel = 0;

  switch ( kid ) {

    //
    // Basic kernels...
    //
    case Basic_MULADDSUB : {
       kernel = new basic::MULADDSUB(run_params);
       break;
    }
    case Basic_IF_QUAD : {
       kernel = new basic::IF_QUAD(run_params);
       break;
    }
    case Basic_TRAP_INT : {
       kernel = new basic::TRAP_INT(run_params);
       break;
    }
    case Basic_INIT3 : {
       kernel = new basic::INIT3(run_params);
       break;
    }
    case Basic_REDUCE3_INT : {
       kernel = new basic::REDUCE3_INT(run_params);
       break;
    }
    case Basic_NESTED_INIT : {
       kernel = new basic::NESTED_INIT(run_params);
       break;
    }

//
// Lcals kernels...
//
#if 0
  Lcals_HYDRO_1D
#endif
    case Lcals_EOS : {
       kernel = new lcals::EOS(run_params);
       break;
    }
#if 0
  Lcals_PIC_2D
  Lcals_DISC_ORD
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
    case Stream_COPY : {
       kernel = new stream::COPY(run_params);
       break;
    }
    case Stream_MUL : {
       kernel = new stream::MUL(run_params);
       break;
    }
    case Stream_ADD : {
       kernel = new stream::ADD(run_params);
       break;
    }
    case Stream_TRIAD : {
       kernel = new stream::TRIAD(run_params);
       break;
    }
    case Stream_DOT : {
       kernel = new stream::DOT(run_params);
       break;
    }

//
// Apps kernels...
//
    case Apps_PRESSURE : {
       kernel = new apps::PRESSURE(run_params);
       break;
    }
    case Apps_ENERGY : {
       kernel = new apps::ENERGY(run_params);
       break;
    }
    case Apps_VOL3D : {
       kernel = new apps::VOL3D(run_params);
       break;
    }
    case Apps_DEL_DOT_VEC_2D : {
       kernel = new apps::DEL_DOT_VEC_2D(run_params);
       break;
    }
    case Apps_COUPLE : {
       kernel = new apps::COUPLE(run_params);
       break;
    }
    case Apps_FIR : {
       kernel = new apps::FIR(run_params);
       break;
    }

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
