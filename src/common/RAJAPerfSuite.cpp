//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
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
#include "basic/INIT_VIEW1D.hpp"
#include "basic/INIT_VIEW1D_OFFSET.hpp"

//
// Lcals kernels...
//
#include "lcals/HYDRO_1D.hpp"
#include "lcals/EOS.hpp"
#include "lcals/INT_PREDICT.hpp"
#include "lcals/DIFF_PREDICT.hpp"
#include "lcals/FIRST_DIFF.hpp"
#include "lcals/PLANCKIAN.hpp"

//
// Polybench kernels...
#include "polybench/POLYBENCH_2MM.hpp"
#include "polybench/POLYBENCH_3MM.hpp"
#include "polybench/POLYBENCH_GEMMVER.hpp"

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
#include "apps/FIR.hpp"
#include "apps/LTIMES.hpp"
#include "apps/LTIMES_NOVIEW.hpp"
#include "apps/WIP-COUPLE.hpp"


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
  std::string("Basic_INIT_VIEW1D"),
  std::string("Basic_INIT_VIEW1D_OFFSET"),

//
// Lcals kernels...
//
  std::string("Lcals_HYDRO_1D"),
  std::string("Lcals_EOS"),
  std::string("Lcals_INT_PREDICT"),
  std::string("Lcals_DIFF_PREDICT"),
  std::string("Lcals_FIRST_DIFF"),
  std::string("Lcals_PLANCKIAN"),

//
// Polybench kernels...
//
  std::string("Polybench_2MM"),
  std::string("Polybench_3MM"),
  std::string("Polybench_GEMMVER"),

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
  std::string("Apps_FIR"),
  std::string("Apps_LTIMES"),
  std::string("Apps_LTIMES_NOVIEW"),
  std::string("Apps_COUPLE"),

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

#if defined(RAJA_ENABLE_OPENMP)
  std::string("Base_OpenMP"),
  std::string("RAJA_OpenMP"),

#if defined(RAJA_ENABLE_TARGET_OPENMP)  
  std::string("Base_OpenMPTarget"),
  std::string("RAJA_OpenMPTarget"),
#endif

#endif

#if defined(RAJA_ENABLE_CUDA)
  std::string("Base_CUDA"),
  std::string("RAJA_CUDA"),
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
    case Basic_INIT_VIEW1D : {
       kernel = new basic::INIT_VIEW1D(run_params);
       break;
    }
    case Basic_INIT_VIEW1D_OFFSET : {
       kernel = new basic::INIT_VIEW1D_OFFSET(run_params);
       break;
    }

//
// Lcals kernels...
//
    case Lcals_HYDRO_1D : {
       kernel = new lcals::HYDRO_1D(run_params);
       break;
    }
    case Lcals_EOS : {
       kernel = new lcals::EOS(run_params);
       break;
    }
    case Lcals_INT_PREDICT : {
       kernel = new lcals::INT_PREDICT(run_params);
       break;
    }
    case Lcals_DIFF_PREDICT : {
       kernel = new lcals::DIFF_PREDICT(run_params);
       break;
    }
    case Lcals_FIRST_DIFF : {
       kernel = new lcals::FIRST_DIFF(run_params);
       break;
    }
    case Lcals_PLANCKIAN : {
       kernel = new lcals::PLANCKIAN(run_params);
       break;
    }

//
// Polybench kernels...
//
    case Polybench_2MM : {
       kernel = new polybench::POLYBENCH_2MM(run_params);
       break;
    }
    case Polybench_3MM : {
       kernel = new polybench::POLYBENCH_3MM(run_params);
       break;
    }
    case Polybench_GEMMVER : {
       kernel = new polybench::POLYBENCH_GEMMVER(run_params);
       break;
    }

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
    case Apps_FIR : {
       kernel = new apps::FIR(run_params);
       break;
    }
    case Apps_LTIMES : {
       kernel = new apps::LTIMES(run_params);
       break;
    }
    case Apps_LTIMES_NOVIEW : {
       kernel = new apps::LTIMES_NOVIEW(run_params);
       break;
    }
    case Apps_COUPLE : {
       kernel = new apps::COUPLE(run_params);
       break;
    }

    default: {
      std::cout << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}

}  // closing brace for rajaperf namespace
