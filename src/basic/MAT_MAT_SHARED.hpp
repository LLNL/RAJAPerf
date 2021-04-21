//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Matrix matrix multiplication with shared memory
///
///

#ifndef RAJAPerf_Basic_MAT_MAT_SHARED_HPP
#define RAJAPerf_Basic_MAT_MAT_SHARED_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

#define TL_SZ 16

#define MAT_MAT_SHARED_DATA_SETUP                                              \
  Real_ptr A = m_A;                                                            \
  Real_ptr B = m_B;                                                            \
  Real_ptr C = m_C;

#define MAT_MAT_SHARED_BODY_0                                                  \
  RAJA_TEAM_SHARED double As[TL_SZ][TL_SZ];                                    \
  RAJA_TEAM_SHARED double Bs[TL_SZ][TL_SZ];                                    \
  RAJA_TEAM_SHARED double Cs[TL_SZ][TL_SZ];

#define MAT_MAT_SHARED_BODY_1 Cs[ty][tx] = 0;

#define MAT_MAT_SHARED_BODY_2                                                  \
  const int Row = by * TL_SZ + ty;                                             \
  const int Col = bx * TL_SZ + tx;                                             \
  if (k * TL_SZ + tx < N && Row < N)                                           \
    As[ty][tx] = A[Row * N + k * TL_SZ + tx];                                  \
  else                                                                         \
    As[ty][tx] = 0.0;                                                          \
  if (k * TL_SZ + ty < N && Col < N)                                           \
    Bs[ty][tx] = B[(k * TL_SZ + ty) * N + Col];                                \
  else                                                                         \
    Bs[ty][tx] = 0.0;

#define MAT_MAT_SHARED_BODY_3                                                  \
  for (int n = 0; n < TL_SZ; ++n)                                              \
    Cs[ty][tx] += As[ty][n] * Bs[n][tx];

#define MAT_MAT_SHARED_BODY_4                                                  \
  const int Row = by * TL_SZ + ty;                                             \
  const int Col = bx * TL_SZ + tx;                                             \
  if (Row < N && Col < N)                                                      \
    C[Col + N * Row] = Cs[ty][tx];

using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
#if defined(RAJA_ENABLE_CUDA)
                                               ,
                                               RAJA::expt::cuda_launch_t<true>
#endif
#if defined(RAJA_ENABLE_HIP)
                                               ,
                                               RAJA::expt::hip_launch_t<true>
#endif
                                               >;

using omp_launch_policy =
    RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t
#if defined(RAJA_ENABLE_CUDA)
                             ,
                             RAJA::expt::cuda_launch_t<true>
#endif
#if defined(RAJA_ENABLE_HIP)
                             ,
                             RAJA::expt::hip_launch_t<true>
#endif
                             >;

using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_CUDA)
using gpu_block_x_policy = RAJA::cuda_block_x_direct;
using gpu_block_y_policy = RAJA::cuda_block_y_direct;
using gpu_thread_x_policy = RAJA::cuda_thread_x_direct;
using gpu_thread_y_policy = RAJA::cuda_thread_y_direct;
#endif

#if defined(RAJA_ENABLE_HIP)
using gpu_block_x_policy = RAJA::hip_block_x_direct;
using gpu_block_y_policy = RAJA::hip_block_y_direct;
using gpu_thread_x_policy = RAJA::hip_thread_x_direct;
using gpu_thread_y_policy = RAJA::hip_thread_y_direct;
#endif

using teams_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_x_policy
#endif
                                       >;

using teams_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_y_policy
#endif
                                       >;

using threads_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                         ,
                                         gpu_thread_x_policy
#endif
                                         >;

using threads_y = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                         ,
                                         gpu_thread_y_policy
#endif
                                         >;

#if 0 // TODO Enable once we update RAJA
using omp_teams = RAJA::expt::LoopPolicy<RAJA::omp_for_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                       ,
                                       gpu_block_y_policy
#endif
                                       >;
#endif

namespace rajaperf {
class RunParams;

namespace basic {

class MAT_MAT_SHARED : public KernelBase {
public:
  MAT_MAT_SHARED(const RunParams &params);

  ~MAT_MAT_SHARED();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
