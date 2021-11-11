//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "PerfsuiteKernelDefinitions.hpp"
#endif

#include <iostream>

namespace rajaperf {

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
    static const std::string GroupNames[] =
            {
                    std::string("Basic"),
                    std::string("Lcals"),
                    std::string("Polybench"),
                    std::string("Stream"),
                    std::string("Apps"),
                    std::string("Algorithm"),

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
    static const std::string KernelNames[] =
            {

// Basic kernels...
//
  std::string("Basic_DAXPY"),
  std::string("Basic_IF_QUAD"),
  std::string("Basic_INIT3"),
  std::string("Basic_INIT_VIEW1D"),
  std::string("Basic_INIT_VIEW1D_OFFSET"),
  std::string("Basic_MAT_MAT_SHARED"),
  std::string("Basic_MULADDSUB"),
  std::string("Basic_NESTED_INIT"),
  std::string("Basic_PI_ATOMIC"),
  std::string("Basic_PI_REDUCE"),
  std::string("Basic_REDUCE3_INT"),
  std::string("Basic_TRAP_INT"),

//
// Lcals kernels...
////
  std::string("Lcals_DIFF_PREDICT"),
  std::string("Lcals_EOS"),
  std::string("Lcals_FIRST_DIFF"),
  std::string("Lcals_FIRST_MIN"),
  std::string("Lcals_FIRST_SUM"),
  std::string("Lcals_GEN_LIN_RECUR"),
  std::string("Lcals_HYDRO_1D"),
  std::string("Lcals_HYDRO_2D"),
  std::string("Lcals_INT_PREDICT"),
  std::string("Lcals_PLANCKIAN"),
  std::string("Lcals_TRIDIAG_ELIM"),
//
//// Polybench kernels...
////
//  std::string("Polybench_2MM"),
//  std::string("Polybench_3MM"),
//  std::string("Polybench_ADI"),
//  std::string("Polybench_ATAX"),
//  std::string("Polybench_FDTD_2D"),
//  std::string("Polybench_FLOYD_WARSHALL"),
//  std::string("Polybench_GEMM"),
//  std::string("Polybench_GEMVER"),
//  std::string("Polybench_GESUMMV"),
//  std::string("Polybench_HEAT_3D"),
//  std::string("Polybench_JACOBI_1D"),
//  std::string("Polybench_JACOBI_2D"),
//  std::string("Polybench_MVT"),
//
////
//// Stream kernels...
////
  std::string("Stream_ADD"),
  std::string("Stream_COPY"),
  std::string("Stream_DOT"),
  std::string("Stream_MUL"),
  std::string("Stream_TRIAD"),
//
// Apps kernels...
//
//  std::string("Apps_COUPLE"),
  std::string("Apps_DEL_DOT_VEC_2D"),
  std::string("Apps_DIFFUSION3DPA"),
  std::string("Apps_ENERGY"),
  std::string("Apps_FIR"),
  std::string("Apps_HALOEXCHANGE"),
  std::string("Apps_HALOEXCHANGE_FUSED"),
  std::string("Apps_LTIMES"),
  std::string("Apps_LTIMES_NOVIEW"),
  std::string("Apps_MASS3DPA"),
  std::string("Apps_PRESSURE"),
  std::string("Apps_VOL3D"),

// Algorithm kernels...
//
  std::string("Algorithm_SORT"),
  std::string("Algorithm_SORTPAIRS"),

                    std::string("Unknown Kernel")  // Keep this at the end and DO NOT remove....

            }; // END KernelNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each VARIANT in suite.
 *
 * IMPORTANT: This is only modified when a new variant is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF VARIANT IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
    static const std::string VariantNames[] =
            {

                    std::string("Base_Seq"),
                    std::string("Lambda_Seq"),
                    std::string("RAJA_Seq"),

                    std::string("Base_OpenMP"),
                    std::string("Lambda_OpenMP"),
                    std::string("RAJA_OpenMP"),

                    std::string("Base_OMPTarget"),
                    std::string("RAJA_OMPTarget"),

  std::string("Base_CUDA"),
  std::string("Lambda_CUDA"),
  std::string("RAJA_CUDA"),

  std::string("Base_HIP"),
  std::string("Lambda_HIP"),
  std::string("RAJA_HIP"),

                    std::string("Kokkos_Lambda"),
                    std::string("Kokkos_Functor"),

                    std::string("Unknown Variant")  // Keep this at the end and DO NOT remove....

            }; // END VariantNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each (RAJA) FEATURE used in suite.
 *
 * IMPORTANT: This is only modified when a new feature is used in suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ENUM OF FEATURE IDS IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string FeatureNames [] =
{

  std::string("Forall"),
  std::string("Kernel"),
  std::string("Teams"),

  std::string("Sort"),
  std::string("Scan"),
  std::string("Workgroup"),

  std::string("Reduction"),
  std::string("Atomic"),

  std::string("View"),

  std::string("Unknown Feature")  // Keep this at the end and DO NOT remove....

}; // END FeatureNames


/*
 *******************************************************************************
 *
 * Return group name associated with GroupID enum value.
 *
 *******************************************************************************
 */
const std::string& getGroupName(GroupID gid)
{
  return GroupNames[gid];
}


/*
 *******************************************************************************
 *
 * Return kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
    std::string getKernelName(KernelID kid) {
        std::string::size_type pos = KernelNames[kid].find("_");
        std::string kname(KernelNames[kid].substr(pos + 1, std::string::npos));
        return kname;
    }


/*
 *******************************************************************************
 *
 * Return full kernel name associated with KernelID enum value.
 *
 *******************************************************************************
 */
    const std::string &getFullKernelName(KernelID kid) {
        return KernelNames[kid];
    }


/*
 *******************************************************************************
 *
 * Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
    const std::string &getVariantName(VariantID vid) {
        return VariantNames[vid];
    }

/*!
 *******************************************************************************
 *
 * Return true if variant associated with VariantID enum value is available 
 * to run; else false.
 *
 *******************************************************************************
 */
    bool isVariantAvailable(VariantID vid) {
        bool ret_val = false;

        if (vid == Base_Seq) {
            ret_val = true;
        }
#if defined(RUN_RAJA_SEQ)
        if (vid == Lambda_Seq ||
            vid == RAJA_Seq) {
            ret_val = true;
        }
#endif

#if defined(RUN_KOKKOS) or defined(RAJAPERF_INFRASTRUCTURE_ONLY)
        if (vid == Kokkos_Lambda ||
            vid == Kokkos_Functor) {
            ret_val = true;
        }
#endif // RUN_KOKKOS

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
        if ( vid == Base_OpenMP ||
             vid == Lambda_OpenMP ||
             vid == RAJA_OpenMP ) {
          ret_val = true;
        }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
        if ( vid == Base_OpenMPTarget ||
             vid == RAJA_OpenMPTarget ) {
          ret_val = true;
        }
#endif

#if defined(RAJA_ENABLE_CUDA)
  if ( vid == Base_CUDA ||
       vid == Lambda_CUDA ||
       vid == RAJA_CUDA ) {
    ret_val = true;
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  if ( vid == Base_HIP ||
       vid == Lambda_HIP ||
       vid == RAJA_HIP ) {
    ret_val = true;
  }
#endif

        return ret_val;
    }

/*
 *******************************************************************************
 *
 * Return feature name associated with FeatureID enum value.
 *
 *******************************************************************************
 */
const std::string& getFeatureName(FeatureID fid)
{
  return FeatureNames[fid];
}

/*
 *******************************************************************************
 *
 * \brief Construct and return kernel object for given KernelID enum value.
 *
 *******************************************************************************
 */
}  // closing brace for rajaperf namespace
