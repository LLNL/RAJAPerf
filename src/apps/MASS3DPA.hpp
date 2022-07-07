//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D mass matrix via partial assembly
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation
/// https://github.com/mfem/mfem/blob/master/fem/bilininteg_mass_pa.cpp#L925
///
/// for (int e = 0; e < NE; ++e) {
///
///   constexpr int MQ1 = MPA_Q1D;
///   constexpr int MD1 = MPA_D1D;
///   constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
///   double sDQ[MQ1 * MD1];
///   double(*Bsmem)[MD1] = (double(*)[MD1])sDQ;
///   double(*Btsmem)[MQ1] = (double(*)[MQ1])sDQ;
///   double sm0[MDQ * MDQ * MDQ];
///   double sm1[MDQ * MDQ * MDQ];
///   double(*Xsmem)[MD1][MD1] = (double(*)[MD1][MD1])sm0;
///   double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1;
///   double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0;
///   double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1;
///   double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0;
///   double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;
///
///   for(int dy=0; dy<MPA_D1D; ++dy) {
///     for(int dx=0; dx<MPA_D1D; ++dx) {
///       for (int dz = 0; dz< MPA_D1D; ++dz) {
///         Xsmem[dz][dy][dx] = X_(dx, dy, dz, e);
///       }
///     }
///     for(int dx=0; dx<MPA_Q1D; ++dx) {
///      Bsmem[dx][dy] = B_(dx, dy);
///     }
///   }
///
///   for(int dy=0; dy<MPA_D1D; ++dy) {
///     for(int dx=0; dx<MPA_Q1D; ++dx) {
///       double u[MPA_D1D];
///       for (int dz = 0; dz < MPA_D1D; dz++) {
///           u[dz] = 0;
///       }
///       for (int dx = 0; dx < MPA_D1D; ++dx) {
///         for (int dz = 0; dz < MPA_D1D; ++dz) {
///           u[dz] += Xsmem[dz][dy][dx] * Bsmem[qx][dx];
///          }
///       }
///       for (int dz = 0; dz < MPA_D1D; ++dz) {
///         DDQ[dz][dy][qx] = u[dz];
///       }
///     }
///   }
///
///   for(int qy=0; qy<MPA_Q1D; ++qy) {
///     for(int qx=0; qx<MPA_Q1D; ++qx) {
///       double u[MPA_D1D];
///       for (int dz = 0; dz < MPA_D1D; dz++) {
///         u[dz] = 0;
///       }
///       for (int dy = 0; dy < MPA_D1D; ++dy) {
///         for (int dz = 0; dz < MPA_D1D; dz++) {
///           u[dz] += DDQ[dz][dy][qx] * Bsmem[qy][dy];
///         }
///       }
///       for (int dz = 0; dz < MPA_D1D; dz++) {
///         DQQ[dz][qy][qx] = u[dz];
///       }
///     }
///   }
///
///   for(int qy=0; qy<MPA_Q1D; ++qy) {
///     for(int qx=0; qx<MPA_Q1D; ++qx) {
///       double u[MPA_Q1D];
///       for (int qz = 0; qz < MPA_Q1D; qz++) {
///         u[qz] = 0;
///       }
///       for (int dz = 0; dz < MPA_D1D; ++dz) {
///         for (int qz = 0; qz < MPA_Q1D; qz++) {
///            u[qz] += DQQ[dz][qy][qx] * Bsmem[qz][dz];
///          }
///       }
///       for (int qz = 0; qz < MPA_Q1D; qz++) {
///         QQQ[qz][qy][qx] = u[qz] * D_(qx, qy, qz, e);
///       }
///     }
///   }
///
///   for(int d=0; d<MPA_D1D; ++d) {
///     for(int q=0; q<MPA_Q1D; ++q) {
///       Btsmem[d][q] = Bt_(q, d);
///     }
///   }
///
///   for(int qy=0; qy<MPA_Q1D; ++qy) {
///     for(int dx=0; dx<MPA_D1D; ++dx) {
///       double u[MPA_Q1D];
///       for (int qz = 0; qz < MPA_Q1D; ++qz) {
///         u[qz] = 0;
///       }
///       for (int qx = 0; qx < MPA_Q1D; ++qx) {
///         for (int qz = 0; qz < MPA_Q1D; ++qz) {
///           u[qz] += QQQ[qz][qy][qx] * Btsmem[dx][qx];
///         }
///       }
///       for (int qz = 0; qz < MPA_Q1D; ++qz) {
///          QQD[qz][qy][dx] = u[qz];
///       }
///     }
///   }
///
///   for(int dy=0; dy<MPA_D1D; ++dy) {
///     for(int dx=0; dx<MPA_D1D; ++dx) {
///       double u[MPA_Q1D];
///       for (int qz = 0; qz < MPA_Q1D; ++qz) {
///          u[qz] = 0;
///       }
///       for (int qy = 0; qy < MPA_Q1D; ++qy) {
///         for (int qz = 0; qz < MPA_Q1D; ++qz) {
///           u[qz] += QQD[qz][qy][dx] * Btsmem[dy][qy];
///          }
///       }
///       for (int qz = 0; qz < MPA_Q1D; ++qz) {
///         QDD[qz][dy][dx] = u[qz];
///       }
///     }
///   }
///
///   for(int dy=0; dy<MPA_D1D; ++dy) {
///     for(int dx=0; dx<MPA_D1D; ++dx) {
///       double u[MPA_D1D];
///       for (int dz = 0; dz < MPA_D1D; ++dz) {
///        u[dz] = 0;
///       }
///       for (int qz = 0; qz < MPA_Q1D; ++qz) {
///         for (int dz = 0; dz < MPA_D1D; ++dz) {
///            u[dz] += QDD[qz][dy][dx] * Btsmem[dz][qz];
///          }
///       }
///       for (int dz = 0; dz < MPA_D1D; ++dz) {
///         Y_(dx, dy, dz, e) += u[dz];
///       }
///     }
///   }
///
/// } // element loop
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
#include "FEM_MACROS.hpp"

#include "RAJA/RAJA.hpp"

//Number of Dofs/Qpts in 1D
#define MPA_D1D 4
#define MPA_Q1D 5
#define B_(x, y) B[x + MPA_Q1D * y]
#define Bt_(x, y) Bt[x + MPA_D1D * y]
#define X_(dx, dy, dz, e)                                                      \
  X[dx + MPA_D1D * dy + MPA_D1D * MPA_D1D * dz + MPA_D1D * MPA_D1D * MPA_D1D * e]
#define Y_(dx, dy, dz, e)                                                      \
  Y[dx + MPA_D1D * dy + MPA_D1D * MPA_D1D * dz + MPA_D1D * MPA_D1D * MPA_D1D * e]
#define D_(qx, qy, qz, e)                                                      \
  D[qx + MPA_Q1D * qy + MPA_Q1D * MPA_Q1D * qz + MPA_Q1D * MPA_Q1D * MPA_Q1D * e]

#define MASS3DPA_0_CPU           \
        constexpr int MQ1 = MPA_Q1D; \
        constexpr int MD1 = MPA_D1D; \
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
        constexpr int MQ1 = MPA_Q1D; \
        constexpr int MD1 = MPA_D1D; \
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
  RAJAPERF_UNROLL(MD1) \
for (int dz = 0; dz< MPA_D1D; ++dz) { \
Xsmem[dz][dy][dx] = X_(dx, dy, dz, e); \
}

#define MASS3DPA_2 \
  Bsmem[dx][dy] = B_(dx, dy);

// 2 * MPA_D1D * MPA_D1D * MPA_D1D * MPA_Q1D
#define MASS3DPA_3 \
  double u[MPA_D1D]; \
RAJAPERF_UNROLL(MD1) \
for (int dz = 0; dz < MPA_D1D; dz++) { \
u[dz] = 0; \
} \
RAJAPERF_UNROLL(MD1) \
for (int dx = 0; dx < MPA_D1D; ++dx) { \
RAJAPERF_UNROLL(MD1) \
for (int dz = 0; dz < MPA_D1D; ++dz) { \
u[dz] += Xsmem[dz][dy][dx] * Bsmem[qx][dx]; \
} \
} \
RAJAPERF_UNROLL(MD1) \
for (int dz = 0; dz < MPA_D1D; ++dz) { \
DDQ[dz][dy][qx] = u[dz]; \
}

//2 * MPA_D1D * MPA_D1D * MPA_Q1D * MPA_Q1D
#define MASS3DPA_4 \
            double u[MPA_D1D]; \
            RAJAPERF_UNROLL(MD1) \
            for (int dz = 0; dz < MPA_D1D; dz++) { \
              u[dz] = 0; \
            } \
            RAJAPERF_UNROLL(MD1) \
            for (int dy = 0; dy < MPA_D1D; ++dy) { \
              RAJAPERF_UNROLL(MD1) \
              for (int dz = 0; dz < MPA_D1D; dz++) { \
                u[dz] += DDQ[dz][dy][qx] * Bsmem[qy][dy]; \
              } \
            } \
            RAJAPERF_UNROLL(MD1) \
            for (int dz = 0; dz < MPA_D1D; dz++) { \
              DQQ[dz][qy][qx] = u[dz]; \
            }

//2 * MPA_D1D * MPA_Q1D * MPA_Q1D * MPA_Q1D + MPA_Q1D * MPA_Q1D * MPA_Q1D
#define MASS3DPA_5 \
            double u[MPA_Q1D]; \
            RAJAPERF_UNROLL(MQ1) \
            for (int qz = 0; qz < MPA_Q1D; qz++) { \
              u[qz] = 0; \
            } \
            RAJAPERF_UNROLL(MD1) \
            for (int dz = 0; dz < MPA_D1D; ++dz) { \
              RAJAPERF_UNROLL(MQ1) \
              for (int qz = 0; qz < MPA_Q1D; qz++) { \
                u[qz] += DQQ[dz][qy][qx] * Bsmem[qz][dz]; \
              } \
            } \
            RAJAPERF_UNROLL(MQ1) \
            for (int qz = 0; qz < MPA_Q1D; qz++) { \
              QQQ[qz][qy][qx] = u[qz] * D_(qx, qy, qz, e); \
            }

#define MASS3DPA_6 \
  Btsmem[d][q] = Bt_(q, d);

//2 * MPA_Q1D * MPA_Q1D * MPA_Q1D * MPA_D1D
#define MASS3DPA_7 \
  double u[MPA_Q1D]; \
RAJAPERF_UNROLL(MQ1) \
for (int qz = 0; qz < MPA_Q1D; ++qz) { \
  u[qz] = 0; \
 } \
RAJAPERF_UNROLL(MQ1) \
for (int qx = 0; qx < MPA_Q1D; ++qx) { \
  RAJAPERF_UNROLL(MQ1) \
    for (int qz = 0; qz < MPA_Q1D; ++qz) { \
      u[qz] += QQQ[qz][qy][qx] * Btsmem[dx][qx]; \
    } \
 } \
RAJAPERF_UNROLL(MQ1) \
for (int qz = 0; qz < MPA_Q1D; ++qz) { \
  QQD[qz][qy][dx] = u[qz]; \
 }

// 2 * MPA_Q1D * MPA_Q1D * MPA_D1D * MPA_D1D
#define MASS3DPA_8 \
            double u[MPA_Q1D]; \
            RAJAPERF_UNROLL(MQ1) \
            for (int qz = 0; qz < MPA_Q1D; ++qz) { \
              u[qz] = 0; \
            } \
            RAJAPERF_UNROLL(MQ1) \
            for (int qy = 0; qy < MPA_Q1D; ++qy) { \
              RAJAPERF_UNROLL(MQ1) \
              for (int qz = 0; qz < MPA_Q1D; ++qz) { \
                u[qz] += QQD[qz][qy][dx] * Btsmem[dy][qy]; \
              } \
            } \
            RAJAPERF_UNROLL(MQ1) \
            for (int qz = 0; qz < MPA_Q1D; ++qz) { \
              QDD[qz][dy][dx] = u[qz]; \
            }

//2 * MPA_Q1D * MPA_D1D * MPA_D1D * MPA_D1D + MPA_D1D * MPA_D1D * MPA_D1D
#define MASS3DPA_9 \
            double u[MPA_D1D]; \
            RAJAPERF_UNROLL(MD1) \
            for (int dz = 0; dz < MPA_D1D; ++dz) { \
              u[dz] = 0; \
            } \
            RAJAPERF_UNROLL(MQ1) \
            for (int qz = 0; qz < MPA_Q1D; ++qz) { \
              RAJAPERF_UNROLL(MD1) \
              for (int dz = 0; dz < MPA_D1D; ++dz) { \
                u[dz] += QDD[qz][dy][dx] * Btsmem[dz][qz]; \
              } \
            } \
            RAJAPERF_UNROLL(MD1) \
            for (int dz = 0; dz < MPA_D1D; ++dz) { \
              Y_(dx, dy, dz, e) += u[dz]; \
            }


namespace rajaperf
{
class RunParams;

namespace apps
{

class MASS3DPA : public KernelBase
{
public:

  MASS3DPA(const RunParams& params);

  ~MASS3DPA();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = MPA_Q1D * MPA_Q1D;
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

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
