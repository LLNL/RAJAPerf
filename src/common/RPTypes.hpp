//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Basic data types used in the Suite.
///

#ifndef RAJAPerf_RPTypes_HPP
#define RAJAPerf_RPTypes_HPP

#include "RAJA/util/types.hpp"

//
// Only one of the following (double or float) should be defined.
//
#define RP_USE_DOUBLE
//#undef RP_USE_DOUBLE

//#define RP_USE_FLOAT
#undef RP_USE_FLOAT

#define RP_USE_COMPLEX
//#undef RP_USE_COMPLEX

#if defined(RP_USE_COMPLEX)
#include <complex>
#endif


namespace rajaperf
{


/*!
 ******************************************************************************
 *
 * \brief Type used for indexing in all kernel repetition loops.
 *
 * It is volatile to ensure that kernels will not be optimized away by
 * compilers, which can happen in some circumstances.
 *
 ******************************************************************************
 */
using RepIndex_type = volatile int;


/*!
 ******************************************************************************
 *
 * \brief Types used for all kernel loop indexing.
 *
 ******************************************************************************
 */
using Index_type = RAJA::Index_type;
///
using Index_ptr = Index_type*;
///
using Index_ptr_ptr = Index_type**;


/*!
 ******************************************************************************
 *
 * \brief Type used for sizing allocations.
 *
 ******************************************************************************
 */
using Size_type = size_t;


/*!
 ******************************************************************************
 *
 * \brief Integer types used in kernels.
 *
 ******************************************************************************
 */
using Int_type = int;
///
using Int_ptr = Int_type*;
using Int_const_ptr = Int_type const*;
///
using Int_ptr_ptr = Int_type**;

using Char_type = char;
///
using Char_ptr = Char_type*;
using Char_const_ptr = Char_type const*;

using Uchar_type = unsigned char;
///
using Uchar_ptr = Uchar_type*;

using Uint64_type = std::uint64_t ;

/*!
 ******************************************************************************
 *
 * \brief Boolean types used in kernels.
 *
 ******************************************************************************
 */
using Bool_type = bool;


/*!
 ******************************************************************************
 *
 * \brief Type used for all kernel checksums.
 *
 ******************************************************************************
 */
using Checksum_type = long double;
///
#define Checksum_MPI_type MPI_LONG_DOUBLE


/*!
 ******************************************************************************
 *
 * \brief Floating point types used in kernels.
 *
 ******************************************************************************
 */
#if defined(RP_USE_DOUBLE)
///
using Real_type = double;
///
#define Real_MPI_type MPI_DOUBLE

#elif defined(RP_USE_FLOAT)
///
using Real_type = float;
///
#define Real_MPI_type MPI_FLOAT

#else
#error Real_type is undefined!

#endif

using Real_ptr = Real_type*;
using Real_const_ptr = Real_type const *;
///
using Real_ptr_ptr = Real_type**;
using Real_const_ptr_ptr = Real_type const **;

#if defined(RP_USE_COMPLEX)
///
using Complex_type = std::complex<Real_type>;

using Complex_ptr = Complex_type*;
#endif


#define RAJAPERF_STRINGIFY_HELPER(...) #__VA_ARGS__
#define RAJAPERF_STRINGIFY(...) RAJAPERF_STRINGIFY_HELPER(__VA_ARGS__)

#define RAJAPERF_CONCAT_HELPER(a, b) a##b
#define RAJAPERF_CONCAT(a, b) RAJAPERF_CONCAT_HELPER(a, b)

#define RAJAPERF_NAME_PER_LINE(name) RAJAPERF_CONCAT(name, __LINE__)

#ifdef _WIN32
#define RAJAPERF_PRAGMA(x) __pragma(x)
#else
#define RAJAPERF_PRAGMA(x) _Pragma(RAJAPERF_STRINGIFY(x))
#endif

#define RAJAPERF_ADD(lhs, rhs) \
      (lhs) += (rhs)

#define RAJAPERF_ATOMIC_ADD_SEQ(lhs, rhs) \
      (lhs) += (rhs)

#define RAJAPERF_ATOMIC_ADD_OMP(lhs, rhs) \
      RAJAPERF_PRAGMA(omp atomic) \
      (lhs) += (rhs)

#define RAJAPERF_ATOMIC_ADD_CUDA(lhs, rhs) \
      ::atomicAdd(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MIN_CUDA(lhs, rhs) \
      ::atomicMin(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MAX_CUDA(lhs, rhs) \
      ::atomicMax(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_HIP(lhs, rhs) \
      ::atomicAdd(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MIN_HIP(lhs, rhs) \
      ::atomicMin(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MAX_HIP(lhs, rhs) \
      ::atomicMax(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_SYCL(lhs, rhs)      \
      sycl::atomic_ref<std::remove_reference_t<decltype(lhs)>,           \
      sycl::memory_order::relaxed,              \
      sycl::memory_scope::device,               \
      sycl::access::address_space::global_space \
      > atomic_y(lhs);                          \
      atomic_y.fetch_add(rhs);

#define RAJAPERF_ATOMIC_ADD_RAJA_SEQ(lhs, rhs) \
      RAJA::atomicAdd<RAJA::seq_atomic>(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_RAJA_OMP(lhs, rhs) \
      RAJA::atomicAdd<RAJA::omp_atomic>(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_RAJA_CUDA(lhs, rhs) \
      RAJA::atomicAdd<RAJA::cuda_atomic>(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MIN_RAJA_CUDA(lhs, rhs) \
      RAJA::atomicMin<RAJA::cuda_atomic>(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MAX_RAJA_CUDA(lhs, rhs) \
      RAJA::atomicMax<RAJA::cuda_atomic>(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_RAJA_HIP(lhs, rhs) \
      RAJA::atomicAdd<RAJA::hip_atomic>(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MIN_RAJA_HIP(lhs, rhs) \
      RAJA::atomicMin<RAJA::hip_atomic>(&(lhs), (rhs))
#define RAJAPERF_ATOMIC_MAX_RAJA_HIP(lhs, rhs) \
      RAJA::atomicMax<RAJA::hip_atomic>(&(lhs), (rhs))

#define RAJAPERF_ATOMIC_ADD_RAJA_SYCL(lhs, rhs) \
      RAJA::atomicAdd<RAJA::sycl_atomic>(&(lhs), (rhs))

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
