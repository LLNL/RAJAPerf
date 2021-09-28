//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D Mass matrix via partial assembly
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation
/// https://github.com/mfem/mfem/blob/master/fem/bilininteg_diffusion_pa.cpp
///
/// for (int e = 0; e < NE; ++e) {
///
///   constexpr int MQ1 = Q1D;
///   constexpr int MD1 = D1D;
///   constexpr int MDQ = (MQ1 >  ? MQ1 : MD1;
///   double sBG[MQ1*MD1];
///   double (*B)[MD1] = (double (*)[MD1]) sBG;
///   double (*G)[MD1] = (double (*)[MD1]) sBG;
///   double (*Bt)[MQ1] = (double (*)[MQ1]) sBG;
///   double (*Gt)[MQ1] = (double (*)[MQ1]) sBG;
///   double sm0[3][MDQ*MDQ*MDQ];
///   double sm1[3][MDQ*MDQ*MDQ];
///   double (*X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2);
///   double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0);
///   double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1);
///   double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0);
///   double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1);
///   double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2);
///   double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0);
///   double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1);
///   double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2);
///   double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0);
///   double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1);
///   double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2);
///   double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0);
///   double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1);
///   double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);
///
///   for(int dy=0; dy<D1D; ++dy) {
///     for(int dx=0; dx<D1D; ++dx) {
///       for (int dz = 0; dz < D1D; ++dz)
///       {
///         X[dz][dy][dx] = x(dx,dy,dz,e);
///       }
///   }
///     for(int qx=0; qx<Q1D; ++qx)
///     {
///       const int i = qi(qx,dy,Q1D);
///       const int j = dj(qx,dy,D1D);
///       const int k = qk(qx,dy,Q1D);
///       const int l = dl(qx,dy,D1D);
///       B[i][j] = b(qx,dy);
///       G[k][l] = g(qx,dy) * sign(qx,dy);
///     }
///   }
///
///   for(int dy=0; dy<D1D; ++dy) {
///     for(int qx=0; qx<Q1D; ++qx) {
///
///       double u[D1D], v[D1D];
///
///       for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = 0.0; }
///
///       for (int dx = 0; dx < D1D; ++dx)
///       {
///         const int i = qi(qx,dx,Q1D);
///         const int j = dj(qx,dx,D1D);
///         const int k = qk(qx,dx,Q1D);
///         const int l = dl(qx,dx,D1D);
///         const double s = sign(qx,dx);
///
///         for (int dz = 0; dz < D1D; ++dz)
///         {
///           const double coords = X[dz][dy][dx];
///           u[dz] += coords * B[i][j];
///           v[dz] += coords * G[k][l] * s;
///         }
///       }
///
///       for (int dz = 0; dz < D1D; ++dz)
///         {
///           DDQ0[dz][dy][qx] = u[dz];
///           DDQ1[dz][dy][qx] = v[dz];
///         }
///     }
///  }
///
///
///   for(int qy=0; qy<Q1D; ++qy) {
///     for(int qx=0; qx<Q1D; ++qx) {
///
///       double u[D1D], v[D1D], w[D1D];
///
///       for (int dz = 0; dz < D1D; dz++) { u[dz] = v[dz] = w[dz] = 0.0; }
///
///       for (int dy = 0; dy < D1D; ++dy)
///       {
///         const int i = qi(qy,dy,Q1D);
///         const int j = dj(qy,dy,D1D);
///         const int k = qk(qy,dy,Q1D);
///         const int l = dl(qy,dy,D1D);
///         const double s = sign(qy,dy);
///
///         for (int dz = 0; dz < D1D; dz++)
///         {
///           u[dz] += DDQ1[dz][dy][qx] * B[i][j];
///           v[dz] += DDQ0[dz][dy][qx] * G[k][l] * s;
///           w[dz] += DDQ0[dz][dy][qx] * B[i][j];
///         }
///       }
///
///       for (int dz = 0; dz < D1D; dz++)
///       {
///         DQQ0[dz][qy][qx] = u[dz];
///         DQQ1[dz][qy][qx] = v[dz];
///         DQQ2[dz][qy][qx] = w[dz];
///       }
///     }
///  }
///
///
///   for(int qy=0; qy<Q1D; ++qy) {
///     for(int qx=0; qx<Q1D; ++qx) {
///
///       double u[Q1D], v[Q1D], w[Q1D];
///
///       for (int qz = 0; qz < Q1D; qz++) { u[qz] = v[qz] = w[qz] = 0.0; }
///
///       for (int dz = 0; dz < D1D; ++dz)
///       {
///         for (int qz = 0; qz < Q1D; qz++)
///         {
///           const int i = qi(qz,dz,Q1D);
///           const int j = dj(qz,dz,D1D);
///           const int k = qk(qz,dz,Q1D);
///           const int l = dl(qz,dz,D1D);
///           const double s = sign(qz,dz);
///           u[qz] += DQQ0[dz][qy][qx] * B[i][j];
///           v[qz] += DQQ1[dz][qy][qx] * B[i][j];
///           w[qz] += DQQ2[dz][qy][qx] * G[k][l] * s;
///          }
///       }
///
///       for (int qz = 0; qz < Q1D; qz++)
///       {
///         const double O11 = d(qx,qy,qz,0,e);
///         const double O12 = d(qx,qy,qz,1,e);
///         const double O13 = d(qx,qy,qz,2,e);
///         const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
///         const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
///         const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
///         const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
///         const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
///         const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
///         const double gX = u[qz];
///         const double gY = v[qz];
///         const double gZ = w[qz];
///         QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
///         QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
///         QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
///       }
///     }
///   }
///
///
///   for(int d=0; d<D1D; ++d) {
///     for(int q=0; q<Q1D; ++q) {
///
///       const int i = qi(q,d,Q1D);
///       const int j = dj(q,d,D1D);
///       const int k = qk(q,d,Q1D);
///       const int l = dl(q,d,D1D);
///       Bt[j][i] = b(q,d);
///       Gt[l][k] = g(q,d) * sign(q,d);
///     }
///   }
///
///
///   for(int qy=0; qy<Q1D; ++qy) {
///     for(int dx=0; dx<D1D; ++D1D) {
///
///       double u[Q1D], v[Q1D], w[Q1D];
///
///       for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
///
///       for (int qx = 0; qx < Q1D; ++qx)
///       {
///         const int i = qi(qx,dx,Q1D);
///          const int j = dj(qx,dx,D1D);
///          const int k = qk(qx,dx,Q1D);
///          const int l = dl(qx,dx,D1D);
///          const double s = sign(qx,dx);
///
///          for (int qz = 0; qz < Q1D; ++qz)
///          {
///            u[qz] += QQQ0[qz][qy][qx] * Gt[l][k] * s;
///            v[qz] += QQQ1[qz][qy][qx] * Bt[j][i];
///            w[qz] += QQQ2[qz][qy][qx] * Bt[j][i];
///          }
///       }
///
///       for (int qz = 0; qz < Q1D; ++qz)
///       {
///         QQD0[qz][qy][dx] = u[qz];
///         QQD1[qz][qy][dx] = v[qz];
///         QQD2[qz][qy][dx] = w[qz];
///        }
///      }
///   }
///
///
///
///   for(int dy=0; dy<D1D; ++dy) {
///     for(int dx=0; dx<D1D; ++dx) {
///
///       double u[Q1D], v[Q1D], w[Q1D];
///
///       for (int qz = 0; qz < Q1D; ++qz) { u[qz] = v[qz] = w[qz] = 0.0; }
///
///       for (int qy = 0; qy < Q1D; ++qy)
///       {
///         const int i = qi(qy,dy,Q1D);
///         const int j = dj(qy,dy,D1D);
///         const int k = qk(qy,dy,Q1D);
///         const int l = dl(qy,dy,D1D);
///         const double s = sign(qy,dy);
///
///         for (int qz = 0; qz < Q1D; ++qz)
///         {
///           u[qz] += QQD0[qz][qy][dx] * Bt[j][i];
///           v[qz] += QQD1[qz][qy][dx] * Gt[l][k] * s;
///           w[qz] += QQD2[qz][qy][dx] * Bt[j][i];
///         }
///     }
///
///      for (int qz = 0; qz < Q1D; ++qz)
///      {
///        QDD0[qz][dy][dx] = u[qz];
///        QDD1[qz][dy][dx] = v[qz];
///        QDD2[qz][dy][dx] = w[qz];
///      }
///     }
///   }
///
///   for(int dy=0; dy<D1D; ++dy) {
///     for(int dx=0; dx<D1D; ++dx) {
///
///       double u[D1D], v[D1D], w[D1D];
///
///       for (int dz = 0; dz < D1D; ++dz) { u[dz] = v[dz] = w[dz] = 0.0; }
///
///       for (int qz = 0; qz < Q1D; ++qz)
///       {
///
///         for (int dz = 0; dz < D1D; ++dz)
///         {
///           const int i = qi(qz,dz,Q1D);
///           const int j = dj(qz,dz,D1D);
///           const int k = qk(qz,dz,Q1D);
///           const int l = dl(qz,dz,D1D);
///           const double s = sign(qz,dz);
///           u[dz] += QDD0[qz][dy][dx] * Bt[j][i];
///           v[dz] += QDD1[qz][dy][dx] * Bt[j][i];
///           w[dz] += QDD2[qz][dy][dx] * Gt[l][k] * s;
///          }
///        }
///
///       for (int dz = 0; dz < D1D; ++dz)
///       {
///         y(dx,dy,dz,e) += (u[dz] + v[dz] + w[dz]);
///        }
///      }
///    }
///
/// } // element loop
///

#ifndef RAJAPerf_Apps_DIFFUSION3DPA_HPP
#define RAJAPerf_Apps_DIFFUSION3DPA_HPP

#define DIFFUSION3DPA_DATA_SETUP \
Real_ptr B = m_B; \
Real_ptr Bt = m_Bt; \
Real_ptr D = m_D; \
Real_ptr X = m_X; \
Real_ptr Y = m_Y; \
Index_type NE = m_NE;

#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

//Number of Dofs/Qpts in 1D
#define D1D 4
#define Q1D 5
#define B_(x, y) B[x + Q1D * y]
#define Bt_(x, y) Bt[x + D1D * y]
#define X_(dx, dy, dz, e)                                                      \
  X[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define Y_(dx, dy, dz, e)                                                      \
  Y[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define D_(qx, qy, qz, e)                                                      \
  D[qx + Q1D * qy + Q1D * Q1D * qz + Q1D * Q1D * Q1D * e]

#define DIFFUSION3DPA_0_GPU
#define DIFFUSION3DPA_0_CPU
#define DIFFUSION3DPA_1
#define DIFFUSION3DPA_2
#define DIFFUSION3DPA_3
#define DIFFUSION3DPA_4
#define DIFFUSION3DPA_5
#define DIFFUSION3DPA_6
#define DIFFUSION3DPA_7
#define DIFFUSION3DPA_8
#define DIFFUSION3DPA_9

#if defined(RAJA_ENABLE_CUDA)
  using m3d_device_launch = RAJA::expt::cuda_launch_t<true>;
  using m3d_gpu_block_x_policy = RAJA::cuda_block_x_direct;
  using m3d_gpu_thread_x_policy = RAJA::cuda_thread_x_loop;
  using m3d_gpu_thread_y_policy = RAJA::cuda_thread_y_loop;
#endif

#if defined(RAJA_ENABLE_HIP)
  using m3d_device_launch = RAJA::expt::hip_launch_t<true>;
  using m3d_gpu_block_x_policy = RAJA::hip_block_x_direct;
  using m3d_gpu_thread_x_policy = RAJA::hip_thread_x_loop;
  using m3d_gpu_thread_y_policy = RAJA::hip_thread_y_loop;
#endif

namespace rajaperf
{
class RunParams;

namespace apps
{

class DIFFUSION3DPA : public KernelBase
{
public:

  DIFFUSION3DPA(const RunParams& params);

  ~DIFFUSION3DPA();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:

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
