/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef CAMP_DEFINES_HPP
#define CAMP_DEFINES_HPP

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace camp
{

#define CAMP_ALLOW_UNUSED_LOCAL(X) (void)X

#if defined(__clang__)
#define CAMP_COMPILER_CLANG
#elif defined(__INTEL_COMPILER)
#define CAMP_COMPILER_INTEL
#elif defined(__xlc__)
#define CAMP_COMPILER_XLC
#elif defined(__PGI)
#define CAMP_COMPILER_PGI
#elif defined(_WIN32)
#define CAMP_COMPILER_MSVC
#elif defined(__GNUC__)
#define RAJA_COMPILER_GNU
#else
#pragma warn("Unknown compiler!")
#endif

#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
#define CAMP_HAS_CONSTEXPR14
#define CAMP_CONSTEXPR14 constexpr
#else
#define CAMP_CONSTEXPR14
#endif

// define host device macros
#define CAMP_HIP_HOST_DEVICE

#if defined(__CUDACC__)
#define CAMP_DEVICE __device__
#define CAMP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_CUDA 1

#if defined(__NVCC__)
#if defined(_WIN32)  // windows is non-compliant, yay
#define CAMP_SUPPRESS_HD_WARN __pragma(nv_exec_check_disable)
#else
#define CAMP_SUPPRESS_HD_WARN _Pragma("nv_exec_check_disable")
#endif
#else // only nvcc supports this pragma
#define CAMP_SUPPRESS_HD_WARN
#endif

#elif defined(__HIPCC__)
#define CAMP_DEVICE __device__
#define CAMP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_HIP 1
#undef CAMP_HIP_HOST_DEVICE
#define CAMP_HIP_HOST_DEVICE __host__ __device__
#define CAMP_HAVE_HIP 1

#define CAMP_SUPPRESS_HD_WARN

#else
#define CAMP_DEVICE
#define CAMP_HOST_DEVICE
#define CAMP_SUPPRESS_HD_WARN
#endif

#if _OPENMP >= 201511
#define CAMP_HAVE_OMP_OFFLOAD 1
#endif

#if defined(__has_builtin)
#if __has_builtin(__make_integer_seq)
#define CAMP_USE_MAKE_INTEGER_SEQ 1
#endif
#endif

// Types
using idx_t = std::ptrdiff_t;
using nullptr_t = decltype(nullptr);

// Helper macros
// TODO: -> CAMP_MAKE_LAMBDA_CONSUMER
#define CAMP_MAKE_L(X)                                             \
  template <typename Lambda, typename... Rest>                     \
  struct X##_l {                                                   \
    using type = typename X<Lambda::template expr, Rest...>::type; \
  }

}  // namespace camp

#endif
