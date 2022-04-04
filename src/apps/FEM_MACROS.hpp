//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJAPerf_FEM_MACROS_HPP
#define RAJAPerf_FEM_MACROS_HPP

#if defined(USE_RAJAPERF_UNROLL)
// If enabled uses RAJA's RAJA_UNROLL_COUNT which is always on
#define RAJAPERF_UNROLL(N) RAJA_UNROLL_COUNT(N)
#else
#define RAJAPERF_UNROLL(N)
#endif

// Need two different host/device macros due to
// how hipcc/clang works.
// See note in MAT_MAT_SHARED regarding hipcc/clang
// builds.
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
#define GPU_FOREACH_THREAD(i, k, N)                    \
  for (int i = threadIdx.k; i < N; i += blockDim.k)
#endif

#define CPU_FOREACH(i, k, N) for (int i = 0; i < N; i++)

#endif // closing endif for header file include guard
