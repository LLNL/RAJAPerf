//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJAPerfSuite.hpp"

#include "RunParams.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include <mpi.h>
#endif

//
// Basic kernels...
//
#include "basic/ARRAY_OF_PTRS.hpp"
#include "basic/COPY8.hpp"
#include "basic/DAXPY.hpp"
#include "basic/DAXPY_ATOMIC.hpp"
#include "basic/IF_QUAD.hpp"
#include "basic/INDEXLIST.hpp"
#include "basic/INDEXLIST_3LOOP.hpp"
#include "basic/INIT3.hpp"
#include "basic/INIT_VIEW1D.hpp"
#include "basic/INIT_VIEW1D_OFFSET.hpp"
#include "basic/MAT_MAT_SHARED.hpp"
#include "basic/MULADDSUB.hpp"
#include "basic/NESTED_INIT.hpp"
#include "basic/PI_ATOMIC.hpp"
#include "basic/PI_REDUCE.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/REDUCE_STRUCT.hpp"
#include "basic/TRAP_INT.hpp"
#include "basic/MULTI_REDUCE.hpp"

//
// Lcals kernels...
//
#include "lcals/DIFF_PREDICT.hpp"
#include "lcals/EOS.hpp"
#include "lcals/FIRST_DIFF.hpp"
#include "lcals/FIRST_MIN.hpp"
#include "lcals/FIRST_SUM.hpp"
#include "lcals/GEN_LIN_RECUR.hpp"
#include "lcals/HYDRO_1D.hpp"
#include "lcals/HYDRO_2D.hpp"
#include "lcals/INT_PREDICT.hpp"
#include "lcals/PLANCKIAN.hpp"
#include "lcals/TRIDIAG_ELIM.hpp"

//
// Polybench kernels...
//
#include "polybench/POLYBENCH_2MM.hpp"
#include "polybench/POLYBENCH_3MM.hpp"
#include "polybench/POLYBENCH_ADI.hpp"
#include "polybench/POLYBENCH_ATAX.hpp"
#include "polybench/POLYBENCH_FDTD_2D.hpp"
#include "polybench/POLYBENCH_FLOYD_WARSHALL.hpp"
#include "polybench/POLYBENCH_GEMM.hpp"
#include "polybench/POLYBENCH_GEMVER.hpp"
#include "polybench/POLYBENCH_GESUMMV.hpp"
#include "polybench/POLYBENCH_HEAT_3D.hpp"
#include "polybench/POLYBENCH_JACOBI_1D.hpp"
#include "polybench/POLYBENCH_JACOBI_2D.hpp"
#include "polybench/POLYBENCH_MVT.hpp"

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
#include "apps/CONVECTION3DPA.hpp"
#include "apps/DEL_DOT_VEC_2D.hpp"
#include "apps/DIFFUSION3DPA.hpp"
#include "apps/EDGE3D.hpp"
#include "apps/ENERGY.hpp"
#include "apps/FIR.hpp"
#include "apps/LTIMES.hpp"
#include "apps/LTIMES_NOVIEW.hpp"
#include "apps/MASS3DEA.hpp"
#include "apps/MASS3DPA.hpp"
#include "apps/MATVEC_3D_STENCIL.hpp"
#include "apps/NODAL_ACCUMULATION_3D.hpp"
#include "apps/PRESSURE.hpp"
#include "apps/VOL3D.hpp"
#include "apps/ZONAL_ACCUMULATION_3D.hpp"

//
// Algorithm kernels...
//
#include "algorithm/SCAN.hpp"
#include "algorithm/SORT.hpp"
#include "algorithm/SORTPAIRS.hpp"
#include "algorithm/REDUCE_SUM.hpp"
#include "algorithm/MEMSET.hpp"
#include "algorithm/MEMCPY.hpp"
#include "algorithm/ATOMIC.hpp"
#include "algorithm/HISTOGRAM.hpp"

//
// Comm kernels...
//
#include "comm/HALO_PACKING.hpp"
#include "comm/HALO_PACKING_FUSED.hpp"
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include "comm/HALO_SENDRECV.hpp"
#include "comm/HALO_EXCHANGE.hpp"
#include "comm/HALO_EXCHANGE_FUSED.hpp"
#endif


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
 *            ITEMS IN THE GroupID enum IN HEADER FILE!!!
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
  std::string("Algorithm"),
  std::string("Comm"),

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
 *            ITEMS IN THE KernelID enum IN HEADER FILE!!! 
 *
 *******************************************************************************
 */
static const std::string KernelNames [] =
{

//
// Basic kernels...
//
  std::string("Basic_ARRAY_OF_PTRS"),
  std::string("Basic_COPY8"),
  std::string("Basic_DAXPY"),
  std::string("Basic_DAXPY_ATOMIC"),
  std::string("Basic_IF_QUAD"),
  std::string("Basic_INDEXLIST"),
  std::string("Basic_INDEXLIST_3LOOP"),
  std::string("Basic_INIT3"),
  std::string("Basic_INIT_VIEW1D"),
  std::string("Basic_INIT_VIEW1D_OFFSET"),
  std::string("Basic_MAT_MAT_SHARED"),
  std::string("Basic_MULADDSUB"),
  std::string("Basic_NESTED_INIT"),
  std::string("Basic_PI_ATOMIC"),
  std::string("Basic_PI_REDUCE"),
  std::string("Basic_REDUCE3_INT"),
  std::string("Basic_REDUCE_STRUCT"),
  std::string("Basic_TRAP_INT"),
  std::string("Basic_MULTI_REDUCE"),

//
// Lcals kernels...
//
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
// Polybench kernels...
//
  std::string("Polybench_2MM"),
  std::string("Polybench_3MM"),
  std::string("Polybench_ADI"),
  std::string("Polybench_ATAX"),
  std::string("Polybench_FDTD_2D"),
  std::string("Polybench_FLOYD_WARSHALL"),
  std::string("Polybench_GEMM"),
  std::string("Polybench_GEMVER"),
  std::string("Polybench_GESUMMV"),
  std::string("Polybench_HEAT_3D"),
  std::string("Polybench_JACOBI_1D"),
  std::string("Polybench_JACOBI_2D"),
  std::string("Polybench_MVT"),

//
// Stream kernels...
//
  std::string("Stream_ADD"),
  std::string("Stream_COPY"),
  std::string("Stream_DOT"),
  std::string("Stream_MUL"),
  std::string("Stream_TRIAD"),

//
// Apps kernels...
//
  std::string("Apps_CONVECTION3DPA"),
  std::string("Apps_DEL_DOT_VEC_2D"),
  std::string("Apps_DIFFUSION3DPA"),
  std::string("Apps_EDGE3D"),
  std::string("Apps_ENERGY"),
  std::string("Apps_FIR"),
  std::string("Apps_LTIMES"),
  std::string("Apps_LTIMES_NOVIEW"),
  std::string("Apps_MASS3DEA"),
  std::string("Apps_MASS3DPA"),
  std::string("Apps_MATVEC_3D_STENCIL"),
  std::string("Apps_NODAL_ACCUMULATION_3D"),
  std::string("Apps_PRESSURE"),
  std::string("Apps_VOL3D"),
  std::string("Apps_ZONAL_ACCUMULATION_3D"),

//
// Algorithm kernels...
//
  std::string("Algorithm_SCAN"),
  std::string("Algorithm_SORT"),
  std::string("Algorithm_SORTPAIRS"),
  std::string("Algorithm_REDUCE_SUM"),
  std::string("Algorithm_MEMSET"),
  std::string("Algorithm_MEMCPY"),
  std::string("Algorithm_ATOMIC"),
  std::string("Algorithm_HISTOGRAM"),

//
// Comm kernels...
//
  std::string("Comm_HALO_PACKING"),
  std::string("Comm_HALO_PACKING_FUSED"),
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  std::string("Comm_HALO_SENDRECV"),
  std::string("Comm_HALO_EXCHANGE"),
  std::string("Comm_HALO_EXCHANGE_FUSED"),
#endif

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
 *            ITEMS IN THE VariantID enum IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string VariantNames [] =
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

  std::string("Base_SYCL"),
  std::string("RAJA_SYCL"),

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
 *            ITEMS IN THE FeatureID enum IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string FeatureNames [] =
{

  std::string("Forall"),
  std::string("Kernel"),
  std::string("Launch"),

  std::string("Sort"),
  std::string("Scan"),
  std::string("Workgroup"),

  std::string("Reduction"),
  std::string("Atomic"),

  std::string("View"),

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  std::string("MPI"),
#endif

  std::string("Unknown Feature")  // Keep this at the end and DO NOT remove....

}; // END FeatureNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each COMPLEXITY used in suite.
 *
 * IMPORTANT: This is only modified when a new complexity is used in suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ITEMS IN THE Complexity enum IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string ComplexityNames [] =
{
  std::string("N"),

  std::string("NlogN"),

  std::string("N^(3/2)"),

  std::string("N^(2/3)"),

  std::string("Unknown Complexity")  // Keep this at the end and DO NOT remove....

}; // END ComplexityNames


/*!
 *******************************************************************************
 *
 * \brief Array of names for each Memory Space in suite.
 *
 * IMPORTANT: This is only modified when a new memory space is added to the suite.
 *
 *            IT MUST BE KEPT CONSISTENT (CORRESPONDING ONE-TO-ONE) WITH
 *            ITEMS IN THE DataSpace enum IN HEADER FILE!!!
 *
 *******************************************************************************
 */
static const std::string DataSpaceNames [] =
{
  std::string("Host"),

  std::string("Omp"),

  std::string("OmpTarget"),

  std::string("CudaPinned"),
  std::string("CudaManaged"),
  std::string("CudaManagedHostPreferred"),
  std::string("CudaManagedDevicePreferred"),
  std::string("CudaManagedHostPreferredDeviceAccessed"),
  std::string("CudaManagedDevicePreferredHostAccessed"),
  std::string("CudaDevice"),

  std::string("HipHostAdviseFine"),
  std::string("HipHostAdviseCoarse"),
  std::string("HipPinned"),
  std::string("HipPinnedFine"),
  std::string("HipPinnedCoarse"),
  std::string("HipManaged"),
  std::string("HipManagedAdviseFine"),
  std::string("HipManagedAdviseCoarse"),
  std::string("HipDevice"),
  std::string("HipDeviceFine"),

  std::string("SyclPinned"),
  std::string("SyclManaged"),
  std::string("SyclDevice"),

  std::string("Unknown Memory"), // Keep this at the end and DO NOT remove....

  std::string("Copy"),

  std::string("Unknown Memory")  // Keep this at the end and DO NOT remove....

}; // END VariantNames


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
std::string getKernelName(KernelID kid)
{
  std::string::size_type pos = KernelNames[kid].find("_");
  std::string kname(KernelNames[kid].substr(pos+1, std::string::npos));
  return kname;
}


/*
 *******************************************************************************
 *
 * Return full kernel name associated with KernelID enum value.
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
 * Return variant name associated with VariantID enum value.
 *
 *******************************************************************************
 */
const std::string& getVariantName(VariantID vid)
{
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
bool isVariantAvailable(VariantID vid)
{
  bool ret_val = false;

  if ( vid == Base_Seq ) {
    ret_val = true;
  }
#if defined(RUN_RAJA_SEQ)
  if ( vid == Lambda_Seq ||
       vid == RAJA_Seq ) {
    ret_val = true;
  }
#endif

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

#if defined(RUN_KOKKOS)
  if ( vid == Kokkos_Lambda ) {
    ret_val = true;
  }
#endif

#if defined(RAJA_ENABLE_SYCL)
  if ( vid == Base_SYCL ||
       vid == RAJA_SYCL ) {
    ret_val = true;
  }
#endif

  return ret_val;
}

/*!
 *******************************************************************************
 *
 * Return true if variant associated with VariantID enum value runs on the GPU.
 *
 *******************************************************************************
 */
bool isVariantGPU(VariantID vid)
{
  bool ret_val = false;

  if ( vid == Base_Seq ) {
    ret_val = false;
  }
#if defined(RUN_RAJA_SEQ)
  if ( vid == Lambda_Seq ||
       vid == RAJA_Seq ) {
    ret_val = false;
  }
#endif

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  if ( vid == Base_OpenMP ||
       vid == Lambda_OpenMP ||
       vid == RAJA_OpenMP ) {
    ret_val = false;
  }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  if ( vid == Base_OpenMPTarget ||
       vid == RAJA_OpenMPTarget ) {
    ret_val = false;
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

#if defined(RUN_KOKKOS)
  if ( vid == Kokkos_Lambda ) {
    ret_val = true;
  }
#endif

#if defined(RAJA_ENABLE_SYCL)
  if ( vid == Base_SYCL ||
       vid == RAJA_SYCL ) {
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
 * Return complexity name associated with Complexity enum value.
 *
 *******************************************************************************
 */
const std::string& getComplexityName(Complexity ac)
{
  return ComplexityNames[static_cast<int>(ac)];
}


/*
 *******************************************************************************
 *
 * Return memory space name associated with DataSpace enum value.
 *
 *******************************************************************************
 */
const std::string& getDataSpaceName(DataSpace ds)
{
  return DataSpaceNames[static_cast<int>(ds)];
}

/*!
 *******************************************************************************
 *
 * Return true if the allocator associated with DataSpace enum value is available.
 *
 *******************************************************************************
 */
bool isDataSpaceAvailable(DataSpace dataSpace)
{
  bool ret_val = false;

  switch (dataSpace) {

    case DataSpace::Host: {
      ret_val = true;
      break;
    }

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case DataSpace::Omp: {
      ret_val = true;
      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case DataSpace::OmpTarget: {
      ret_val = true;
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case DataSpace::CudaPinned:
    case DataSpace::CudaManaged:
    case DataSpace::CudaManagedHostPreferred:
    case DataSpace::CudaManagedDevicePreferred:
    case DataSpace::CudaManagedHostPreferredDeviceAccessed:
    case DataSpace::CudaManagedDevicePreferredHostAccessed:
    case DataSpace::CudaDevice: {
      ret_val = true;
      break;
    }
#endif

#if defined(RAJA_ENABLE_HIP)
    case DataSpace::HipHostAdviseFine:
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipHostAdviseCoarse:
#endif
    case DataSpace::HipPinned:
    case DataSpace::HipPinnedFine:
    case DataSpace::HipPinnedCoarse:
    case DataSpace::HipManaged:
    case DataSpace::HipManagedAdviseFine:
#if defined(RAJAPERF_USE_MEMADVISE_COARSE)
    case DataSpace::HipManagedAdviseCoarse:
#endif
    case DataSpace::HipDevice:
    case DataSpace::HipDeviceFine: {
      ret_val = true;
      break;
    } 
#endif

#if defined(RAJA_ENABLE_SYCL)
    case DataSpace::SyclPinned:
    case DataSpace::SyclManaged:
    case DataSpace::SyclDevice: {
      ret_val = true;
      break;
    }
#endif

    default: {
      ret_val = false;
      break;
    }

  } // close switch (dataSpace)

  return ret_val;
}

/*!
 *******************************************************************************
 *
 * Return true if the DataSpace enum value is a psuedo DataSpace.
 *
 *******************************************************************************
 */
bool isPseudoDataSpace(DataSpace dataSpace)
{
  bool ret_val = false;

  switch (dataSpace) {

    case DataSpace::Copy: {
      ret_val = true;
      break;
    }
    default: {
      ret_val = false;
      break;
    }

  }

  return ret_val;
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
    case Basic_ARRAY_OF_PTRS : {
       kernel = new basic::ARRAY_OF_PTRS(run_params);
       break;
    }
    case Basic_COPY8 : {
       kernel = new basic::COPY8(run_params);
       break;
    }
    case Basic_DAXPY : {
       kernel = new basic::DAXPY(run_params);
       break;
    }
    case Basic_DAXPY_ATOMIC : {
       kernel = new basic::DAXPY_ATOMIC(run_params);
       break;
    }
    case Basic_IF_QUAD : {
       kernel = new basic::IF_QUAD(run_params);
       break;
    }
    case Basic_INDEXLIST : {
       kernel = new basic::INDEXLIST(run_params);
       break;
    }
    case Basic_INDEXLIST_3LOOP : {
       kernel = new basic::INDEXLIST_3LOOP(run_params);
       break;
    }
    case Basic_INIT3 : {
       kernel = new basic::INIT3(run_params);
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
    case Basic_MAT_MAT_SHARED : {
       kernel = new basic::MAT_MAT_SHARED(run_params);
       break;
    }
    case Basic_MULADDSUB : {
       kernel = new basic::MULADDSUB(run_params);
       break;
    }
    case Basic_NESTED_INIT : {
       kernel = new basic::NESTED_INIT(run_params);
       break;
    }
    case Basic_PI_ATOMIC : {
       kernel = new basic::PI_ATOMIC(run_params);
       break;
    }
    case Basic_PI_REDUCE : {
       kernel = new basic::PI_REDUCE(run_params);
       break;
    }
    case Basic_REDUCE3_INT : {
       kernel = new basic::REDUCE3_INT(run_params);
       break;
    }
    case Basic_REDUCE_STRUCT : { 
        kernel = new basic::REDUCE_STRUCT(run_params);
        break;
    } 	
    case Basic_TRAP_INT : {
       kernel = new basic::TRAP_INT(run_params);
       break;
    }
    case Basic_MULTI_REDUCE : {
       kernel = new basic::MULTI_REDUCE(run_params);
       break;
    }

//
// Lcals kernels...
//
    case Lcals_DIFF_PREDICT : {
       kernel = new lcals::DIFF_PREDICT(run_params);
       break;
    }
    case Lcals_EOS : {
       kernel = new lcals::EOS(run_params);
       break;
    }
    case Lcals_FIRST_DIFF : {
       kernel = new lcals::FIRST_DIFF(run_params);
       break;
    }
    case Lcals_FIRST_MIN : {
       kernel = new lcals::FIRST_MIN(run_params);
       break;
    }
    case Lcals_FIRST_SUM : {
       kernel = new lcals::FIRST_SUM(run_params);
       break;
    }
    case Lcals_GEN_LIN_RECUR : {
       kernel = new lcals::GEN_LIN_RECUR(run_params);
       break;
    }
    case Lcals_HYDRO_1D : {
       kernel = new lcals::HYDRO_1D(run_params);
       break;
    }
    case Lcals_HYDRO_2D : {
       kernel = new lcals::HYDRO_2D(run_params);
       break;
    }
    case Lcals_INT_PREDICT : {
       kernel = new lcals::INT_PREDICT(run_params);
       break;
    }
    case Lcals_PLANCKIAN : {
       kernel = new lcals::PLANCKIAN(run_params);
       break;
    }
    case Lcals_TRIDIAG_ELIM : {
       kernel = new lcals::TRIDIAG_ELIM(run_params);
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
    case Polybench_ADI  : {
       kernel = new polybench::POLYBENCH_ADI(run_params);
       break;
    }
    case Polybench_ATAX  : {
       kernel = new polybench::POLYBENCH_ATAX(run_params);
       break;
    }
    case Polybench_FDTD_2D : {
       kernel = new polybench::POLYBENCH_FDTD_2D(run_params);
       break;
    }
    case Polybench_FLOYD_WARSHALL : {
       kernel = new polybench::POLYBENCH_FLOYD_WARSHALL(run_params);
       break;
    }
    case Polybench_GEMM : {
       kernel = new polybench::POLYBENCH_GEMM(run_params);
       break;
    }
    case Polybench_GEMVER : {
       kernel = new polybench::POLYBENCH_GEMVER(run_params);
       break;
    }
    case Polybench_GESUMMV : {
       kernel = new polybench::POLYBENCH_GESUMMV(run_params);
       break;
    }
    case Polybench_HEAT_3D : {
       kernel = new polybench::POLYBENCH_HEAT_3D(run_params);
       break;
    }
    case Polybench_JACOBI_1D : {
       kernel = new polybench::POLYBENCH_JACOBI_1D(run_params);
       break;
    }
    case Polybench_JACOBI_2D : {
       kernel = new polybench::POLYBENCH_JACOBI_2D(run_params);
       break;
    }
    case Polybench_MVT : {
       kernel = new polybench::POLYBENCH_MVT(run_params);
       break;
    }

//
// Stream kernels...
//
    case Stream_ADD : {
       kernel = new stream::ADD(run_params);
       break;
    }
    case Stream_COPY : {
       kernel = new stream::COPY(run_params);
       break;
    }
    case Stream_DOT : {
       kernel = new stream::DOT(run_params);
       break;
    }
    case Stream_MUL : {
       kernel = new stream::MUL(run_params);
       break;
    }
    case Stream_TRIAD : {
       kernel = new stream::TRIAD(run_params);
       break;
    }

//
// Apps kernels...
//
    case Apps_CONVECTION3DPA : {
       kernel = new apps::CONVECTION3DPA(run_params);
       break;
    }

    case Apps_DEL_DOT_VEC_2D : {
       kernel = new apps::DEL_DOT_VEC_2D(run_params);
       break;
    }
    case Apps_DIFFUSION3DPA : {
       kernel = new apps::DIFFUSION3DPA(run_params);
       break;
    }
    case Apps_EDGE3D : {
       kernel = new apps::EDGE3D(run_params);
       break;
    }
    case Apps_ENERGY : {
       kernel = new apps::ENERGY(run_params);
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
    case Apps_MASS3DEA : {
       kernel = new apps::MASS3DEA(run_params);
       break;
    }      
    case Apps_MASS3DPA : {
       kernel = new apps::MASS3DPA(run_params);
       break;
    }
    case Apps_MATVEC_3D_STENCIL : {
       kernel = new apps::MATVEC_3D_STENCIL(run_params);
       break;
    }
    case Apps_NODAL_ACCUMULATION_3D : {
       kernel = new apps::NODAL_ACCUMULATION_3D(run_params);
       break;
    }
    case Apps_PRESSURE : {
       kernel = new apps::PRESSURE(run_params);
       break;
    }
    case Apps_VOL3D : {
       kernel = new apps::VOL3D(run_params);
       break;
    }
    case Apps_ZONAL_ACCUMULATION_3D : {
       kernel = new apps::ZONAL_ACCUMULATION_3D(run_params);
       break;
    }

//
// Algorithm kernels...
//
    case Algorithm_SCAN: {
       kernel = new algorithm::SCAN(run_params);
       break;
    }
    case Algorithm_SORT: {
       kernel = new algorithm::SORT(run_params);
       break;
    }
    case Algorithm_SORTPAIRS: {
       kernel = new algorithm::SORTPAIRS(run_params);
       break;
    }
    case Algorithm_REDUCE_SUM: {
       kernel = new algorithm::REDUCE_SUM(run_params);
       break;
    }
    case Algorithm_MEMSET: {
       kernel = new algorithm::MEMSET(run_params);
       break;
    }
    case Algorithm_MEMCPY: {
       kernel = new algorithm::MEMCPY(run_params);
       break;
    }
    case Algorithm_ATOMIC: {
       kernel = new algorithm::ATOMIC(run_params);
       break;
    }
    case Algorithm_HISTOGRAM: {
       kernel = new algorithm::HISTOGRAM(run_params);
       break;
    }

//
// Comm kernels...
//
    case Comm_HALO_PACKING : {
       kernel = new comm::HALO_PACKING(run_params);
       break;
    }
    case Comm_HALO_PACKING_FUSED : {
       kernel = new comm::HALO_PACKING_FUSED(run_params);
       break;
    }
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
    case Comm_HALO_SENDRECV : {
       kernel = new comm::HALO_SENDRECV(run_params);
       break;
    }
    case Comm_HALO_EXCHANGE : {
       kernel = new comm::HALO_EXCHANGE(run_params);
       break;
    }
    case Comm_HALO_EXCHANGE_FUSED : {
       kernel = new comm::HALO_EXCHANGE_FUSED(run_params);
       break;
    }
#endif

    default: {
      getCout() << "\n Unknown Kernel ID = " << kid << std::endl;
    }

  } // end switch on kernel id

  return kernel;
}


// subclass of streambuf that ignores overflow
// never printing anything to the underlying stream
struct NullStream : std::streambuf, std::ostream
{
  using Base = std::streambuf;
  using int_type = typename Base::int_type;

  NullStream() : std::ostream(this) {}
public:
  int_type overflow(int_type c) override { return c; }
};

std::ostream* makeNullStream()
{
  return new NullStream();
}

std::ostream& getNullStream()
{
  static NullStream null_stream;
  return null_stream;
}

std::ostream& getCout()
{
  int rank = 0;
#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (rank == 0) {
    return std::cout;
  }
  return getNullStream();
}

}  // closing brace for rajaperf namespace
