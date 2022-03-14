//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJAPerf_FEM_MACROS_HPP
#define RAJAPerf_FEM_MACROS_HPP

#define RAJAPERF_DIRECT_PRAGMA(X) _Pragma(#X)
#if defined(USE_RAJAPERF_UNROLL)
#define RAJAPERF_UNROLL(N) RAJAPERF_DIRECT_PRAGMA(unroll(N))
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
