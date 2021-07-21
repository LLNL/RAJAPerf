//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Matrix matrix multiplication with shared memory
/// reference implementation:
///
///      for (Index_type by = 0; by < Ny; ++by) {
///        for (Index_type bx = 0; bx < Nx; ++bx) {
///
///          double As[TL_SZ][TL_SZ];
///          double Bs[TL_SZ][TL_SZ];
///          double Cs[TL_SZ][TL_SZ];
///
///          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
///            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
///                Cs[ty][tx] = 0;
///            }
///          }
///
///          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {
///
///            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
///              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
///                const Index_type Row = by * TL_SZ + ty;
///                const Index_type Col = bx * TL_SZ + tx;
///                if (k * TL_SZ + tx < N && Row < N)
///                  As[ty][tx] = A[Row * N + k * TL_SZ + tx];
///                else
///                  As[ty][tx] = 0.0;
///                if (k * TL_SZ + ty < N && Col < N)
///                  Bs[ty][tx] = B[(k * TL_SZ + ty) * N + Col];
///                else
///                  Bs[ty][tx] = 0.0;
///              }
///            }
///
///            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
///              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
///                for (Index_type n = 0; n < TL_SZ; ++n)
///                  Cs[ty][tx] += As[ty][n] * Bs[n][tx];
///              }
///            }
///
///          }
///
///          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
///            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
///
///              const Index_type Row = by * TL_SZ + ty;
///              const Index_type Col = bx * TL_SZ + tx;
///              if (Row < N && Col < N)
///                C[Col + N * Row] = Cs[ty][tx];
///            }
///          }
///        }
///      }
///
///

#ifndef RAJAPerf_Basic_MAT_MAT_SHARED_HPP
#define RAJAPerf_Basic_MAT_MAT_SHARED_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

constexpr rajaperf::Index_type TL_SZ = 16;

#define MAT_MAT_SHARED_DATA_SETUP                                              \
  Real_ptr A = m_A;                                                            \
  Real_ptr B = m_B;                                                            \
  Real_ptr C = m_C;

/*
 When doing the device compile pass hipcc/clang will put in the device
 versions of the macros everywhere, in device functions, host device functions,
 and host only functions. Then it will make sure that code is valid everywhere,
 that's fine for device and host device functions, but it is not ok for host only
 functions. Nvcc doesn't look at host only code when it does the device pass
 so it doesn't see these kind of problems.
 */
#define MAT_MAT_SHARED_BODY_0_CLANG_HIP_CPU                   \
  double As[TL_SZ][TL_SZ];                                    \
  double Bs[TL_SZ][TL_SZ];                                    \
  double Cs[TL_SZ][TL_SZ];

#define MAT_MAT_SHARED_BODY_0                                                  \
  RAJA_TEAM_SHARED double As[TL_SZ][TL_SZ];                                    \
  RAJA_TEAM_SHARED double Bs[TL_SZ][TL_SZ];                                    \
  RAJA_TEAM_SHARED double Cs[TL_SZ][TL_SZ];

#define MAT_MAT_SHARED_BODY_1 Cs[ty][tx] = 0;

#define MAT_MAT_SHARED_BODY_2                                                  \
  const Index_type Row = by * TL_SZ + ty;                                      \
  const Index_type Col = bx * TL_SZ + tx;                                      \
  if (k * TL_SZ + tx < N && Row < N)                                           \
    As[ty][tx] = A[Row * N + k * TL_SZ + tx];                                  \
  else                                                                         \
    As[ty][tx] = 0.0;                                                          \
  if (k * TL_SZ + ty < N && Col < N)                                           \
    Bs[ty][tx] = B[(k * TL_SZ + ty) * N + Col];                                \
  else                                                                         \
    Bs[ty][tx] = 0.0;

#define MAT_MAT_SHARED_BODY_3                                                  \
  for (Index_type n = 0; n < TL_SZ; ++n)                                       \
    Cs[ty][tx] += As[ty][n] * Bs[n][tx];

#define MAT_MAT_SHARED_BODY_4                                                  \
  const Index_type Row = by * TL_SZ + ty;                                      \
  const Index_type Col = bx * TL_SZ + tx;                                      \
  if (Row < N && Col < N)                                                      \
    C[Col + N * Row] = Cs[ty][tx];

#if defined(RAJA_ENABLE_CUDA)
    using mms_device_launch = RAJA::expt::cuda_launch_t<true>;
    using mms_gpu_block_x_policy = RAJA::cuda_block_x_direct;
    using mms_gpu_block_y_policy = RAJA::cuda_block_y_direct;
    using mms_gpu_thread_x_policy = RAJA::cuda_thread_x_direct;
    using mms_gpu_thread_y_policy = RAJA::cuda_thread_y_direct;
#endif

#if defined(RAJA_ENABLE_HIP)
    using mms_device_launch = RAJA::expt::hip_launch_t<true>;
    using mms_gpu_block_x_policy = RAJA::hip_block_x_direct;
    using mms_gpu_block_y_policy = RAJA::hip_block_y_direct;
    using mms_gpu_thread_x_policy = RAJA::hip_thread_x_direct;
    using mms_gpu_thread_y_policy = RAJA::hip_thread_y_direct;
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
  void runStdParVariant(VariantID vid);

private:
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;

  Index_type m_N;
  Index_type m_N_default;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
