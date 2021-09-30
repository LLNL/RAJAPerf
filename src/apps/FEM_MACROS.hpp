//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJAPerf_FEM_MACROS_HPP
#define RAJAPerf_FEM_MACROS_HPP

#define RAJA_DIRECT_PRAGMA(X) _Pragma(#X)
#if defined(USE_RAJA_UNROLL)
#define RAJA_UNROLL(N) RAJA_DIRECT_PRAGMA(unroll(N))
#else
#define RAJA_UNROLL(N)
#endif

#if defined(RAJA_DEVICE_CODE)
#define FOREACH_THREAD(i, k, N)                                                \
  for (int i = threadIdx.k; i < N; i += blockDim.k)
#else
#define FOREACH_THREAD(i, k, N) for (int i = 0; i < N; i++)
#endif

#endif // closing endif for header file include guard
