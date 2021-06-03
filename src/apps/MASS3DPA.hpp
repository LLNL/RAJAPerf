//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D Mass matrix via partial assembly
///
/// TODO add description
///
///

#ifndef RAJAPerf_Apps_MASS3DPA_HPP
#define RAJAPerf_Apps_MASS3DPA_HPP

#define MASS3DPA_DATA_SETUP \
Real_ptr B = m_B; \
Real_ptr Bt = m_Bt; \
Real_ptr D = m_D; \
Real_ptr X = m_X; \
Real_ptr Y = m_Y; \
Index_type NE = m_NE; 

#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#define MASS3DPA_0_CPU \
        constexpr int MQ1 = Q1D; \
        constexpr int MD1 = D1D; \
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1; \
        double sDQ[MQ1 * MD1]; \
        double(*Bsmem)[MD1] = (double(*)[MD1])sDQ; \
        double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ; \
        double sm0[MDQ * MDQ * MDQ]; \
        double sm1[MDQ * MDQ * MDQ]; \
        double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0; \
        double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1; \
        double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0; \
        double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1; \
        double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0; \
        double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

#define MASS3DPA_0_GPU \
        constexpr int MQ1 = Q1D; \
        constexpr int MD1 = D1D; \
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1; \
        RAJA_TEAM_SHARED  double sDQ[MQ1 * MD1];     \
        double(*Bsmem)[MD1] = (double(*)[MD1])sDQ; \
        double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ; \
        RAJA_TEAM_SHARED double sm0[MDQ * MDQ * MDQ];       \
        RAJA_TEAM_SHARED double sm1[MDQ * MDQ * MDQ];      \
        double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0; \
        double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1; \
        double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0; \
        double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1; \
        double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0; \
        double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

#define MASS3DPA_1 \
  RAJA_UNROLL(MD1) \
for (int dz = 0; dz< D1D; ++dz) { \
Xsmem[dz][dy][dx] = X_(dx, dy, dz, e); \
}

#define MASS3DPA_2 \
  Bsmem[dx][dy] = B_(dx, dy);

#define MASS3DPA_3 \
  double u[D1D]; \
RAJA_UNROLL(MD1) \
for (int dz = 0; dz < D1D; dz++) { \
u[dz] = 0; \
} \
RAJA_UNROLL(MD1) \
for (int dx = 0; dx < D1D; ++dx) { \
RAJA_UNROLL(MD1) \
for (int dz = 0; dz < D1D; ++dz) { \
u[dz] += Xsmem[dz][dy][dx] * Bsmem[qx][dx]; \
} \
} \
RAJA_UNROLL(MD1) \
for (int dz = 0; dz < D1D; ++dz) { \
DDQ[dz][dy][qx] = u[dz]; \
}

#define MASS3DPA_4 \
            double u[D1D]; \
            RAJA_UNROLL(MD1) \
            for (int dz = 0; dz < D1D; dz++) { \
              u[dz] = 0; \
            } \
            RAJA_UNROLL(MD1) \
            for (int dy = 0; dy < D1D; ++dy) { \
              RAJA_UNROLL(MD1) \
              for (int dz = 0; dz < D1D; dz++) { \
                u[dz] += DDQ[dz][dy][qx] * Bsmem[qy][dy]; \
              } \
            } \
            RAJA_UNROLL(MD1) \
            for (int dz = 0; dz < D1D; dz++) { \
              DQQ[dz][qy][qx] = u[dz]; \
            }

#define MASS3DPA_5 \
            double u[Q1D]; \
            RAJA_UNROLL(MQ1) \
            for (int qz = 0; qz < Q1D; qz++) { \
              u[qz] = 0; \
            } \
            RAJA_UNROLL(MD1) \
            for (int dz = 0; dz < D1D; ++dz) { \
              RAJA_UNROLL(MQ1) \
              for (int qz = 0; qz < Q1D; qz++) { \
                u[qz] += DQQ[dz][qy][qx] * Bsmem[qz][dz]; \
              } \
            } \
            RAJA_UNROLL(MQ1) \
            for (int qz = 0; qz < Q1D; qz++) { \
              QQQ[qz][qy][qx] = u[qz] * D_(qx, qy, qz, e); \
            } 

#define MASS3DPA_6 \
  Btsmem[d][q] = Bt_(q, d);

#define MASS3DPA_7 \
  double u[Q1D]; \
RAJA_UNROLL(MQ1) \
for (int qz = 0; qz < Q1D; ++qz) { \
  u[qz] = 0; \
 } \
RAJA_UNROLL(MQ1) \
for (int qx = 0; qx < Q1D; ++qx) { \
  RAJA_UNROLL(MQ1) \
    for (int qz = 0; qz < Q1D; ++qz) { \
      u[qz] += QQQ[qz][qy][qx] * Btsmem[dx][qx]; \
    } \
 } \
RAJA_UNROLL(MQ1) \
for (int qz = 0; qz < Q1D; ++qz) { \
  QQD[qz][qy][dx] = u[qz]; \
 }

#define MASS3DPA_8 \
            double u[Q1D]; \
            RAJA_UNROLL(MQ1) \
            for (int qz = 0; qz < Q1D; ++qz) { \
              u[qz] = 0; \
            } \
            RAJA_UNROLL(MQ1) \
            for (int qy = 0; qy < Q1D; ++qy) { \
              RAJA_UNROLL(MQ1) \
              for (int qz = 0; qz < Q1D; ++qz) { \
                u[qz] += QQD[qz][qy][dx] * Btsmem[dy][qy]; \
              } \
            } \
            RAJA_UNROLL(MQ1) \
            for (int qz = 0; qz < Q1D; ++qz) { \
              QDD[qz][dy][dx] = u[qz]; \
            }

#define MASS3DPA_9 \
            double u[D1D]; \
            RAJA_UNROLL(MD1) \
            for (int dz = 0; dz < D1D; ++dz) { \
              u[dz] = 0; \
            } \
            RAJA_UNROLL(MQ1) \
            for (int qz = 0; qz < Q1D; ++qz) { \
              RAJA_UNROLL(MD1) \
              for (int dz = 0; dz < D1D; ++dz) { \
                u[dz] += QDD[qz][dy][dx] * Btsmem[dz][qz]; \
              } \
            } \
            RAJA_UNROLL(MD1) \
            for (int dz = 0; dz < D1D; ++dz) { \
              Y_(dx, dy, dz, e) += u[dz]; \
            }




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

#if defined(RAJA_ENABLE_OPENMP)
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
#endif

  using loop_policy = RAJA::loop_exec;

#if defined(RAJA_ENABLE_CUDA)
  using gpu_block_x_policy = RAJA::cuda_block_x_direct;
  using gpu_thread_x_policy = RAJA::cuda_thread_x_loop;
  using gpu_thread_y_policy = RAJA::cuda_thread_y_loop;
#endif

#if defined(RAJA_ENABLE_HIP)
  using gpu_block_x_policy = RAJA::hip_block_x_direct;
  using gpu_thread_x_policy = RAJA::hip_thread_x_loop;
  using gpu_thread_y_policy = RAJA::hip_thread_y_loop;
#endif

  using teams_x = RAJA::expt::LoopPolicy<loop_policy
#if defined(RAJA_DEVICE_ACTIVE)
                                         ,
                                       gpu_block_x_policy
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
#if defined(RAJA_ENABLE_OPENMP)
  using omp_teams = RAJA::expt::LoopPolicy<RAJA::omp_for_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                       gpu_block_x_policy
#endif
                                           >;
#endif

namespace rajaperf 
{
class RunParams;

namespace apps
{

//
// These index value types cannot be defined in function scope for
// RAJA CUDA variant to work.
//

class MASS3DPA : public KernelBase
{
public:

  MASS3DPA(const RunParams& params);

  ~MASS3DPA();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);

private:

  int m_Q1D = 5; 
  int m_D1D = 4;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_NE;
  Index_type m_NE_default;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
