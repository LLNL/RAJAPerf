//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Action of 3D diffusion matrix via partial assembly
///
/// Based on MFEM's/CEED algorithms.
/// Reference implementation
/// https://github.com/mfem/mfem/blob/master/fem/bilininteg_diffusion_pa.cpp
///
/// for (int e = 0; e < NE; ++e) {
///
///   constexpr int MQ1 = DPA_Q1D;
///   constexpr int MD1 = DPA_D1D;
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
///   for(int dz=0;dz<D1D;dz++){
///     for(int dy=0;dy<D1D;++dy){
///         for(int dx=0; dx<D1D;++dx){
///            X[dz][dy][dx] = x(dx,dy,dz,e);
///         }
///      }
///   }
///
///   for(int dy=0; dy<D1D; ++dy){
///     for(int qx=0; qx<Q1D; ++qx){
///       const int i = qi(qx,dy,Q1D);
///       const int j = dj(qx,dy,D1D);
///       const int k = qk(qx,dy,Q1D);
///       const int l = dl(qx,dy,D1D);
///       B[i][j] = b(qx,dy);
///       G[k][l] = g(qx,dy) * sign(qx,dy);
///     }
///   }
///
///   for(int dz=0;dz<D1D;dz++){
///     for(int dy=0;dy<D1D;++dy){
///       for(int qx=0; qx<Q1D; qx++){
///         double u = 0.0, v = 0.0;
///         for (int dx = 0; dx < D1D; ++dx){
///            const int i = qi(qx,dx,Q1D);
///            const int j = dj(qx,dx,D1D);
///            const int k = qk(qx,dx,Q1D);
///            const int l = dl(qx,dx,D1D);
///            const double s = sign(qx,dx);
///            const double coords = X[dz][dy][dx];
///            u += coords * B[i][j];
///            v += coords * G[k][l] * s;
///         }
///         DDQ0[dz][dy][qx] = u;
///         DDQ1[dz][dy][qx] = v;
///       }
///     }
///   }
///
///    for(int dz=0;dz<D1D;dz++){
///      for(int qy=0;qy<Q1D;++qy){
///         for(int qx=0; qx<Q1D;++qx){
///           double u = 0.0, v = 0.0, w = 0.0;
///           for (int dy = 0; dy < D1D; ++dy){
///             const int i = qi(qy,dy,Q1D);
///             const int j = dj(qy,dy,D1D);
///             const int k = qk(qy,dy,Q1D);
///             const int l = dl(qy,dy,D1D);
///             const double s = sign(qy,dy);
///             u += DDQ1[dz][dy][qx] * B[i][j];
///             v += DDQ0[dz][dy][qx] * G[k][l] * s;
///             w += DDQ0[dz][dy][qx] * B[i][j];
///           }
///           DQQ0[dz][qy][qx] = u;
///           DQQ1[dz][qy][qx] = v;
///           DQQ2[dz][qy][qx] = w;
///         }
///      }
///   }
///
///   for(int qz=0;qz<Q1D;qz++){
///     for(int qy=0;qy<Q1D;++qy){
///       for(int qx=0; qx<Q1D;++qx){
///
///         double u = 0.0, v = 0.0, w = 0.0;
///         for (int dz = 0; dz < D1D; ++dz){
///           const int i = qi(qz,dz,Q1D);
///           const int j = dj(qz,dz,D1D);
///           const int k = qk(qz,dz,Q1D);
///           const int l = dl(qz,dz,D1D);
///           const double s = sign(qz,dz);
///           u += DQQ0[dz][qy][qx] * B[i][j];
///           v += DQQ1[dz][qy][qx] * B[i][j];
///           w += DQQ2[dz][qy][qx] * G[k][l] * s;
///         }
///         const double O11 = d(qx,qy,qz,0,e);
///         const double O12 = d(qx,qy,qz,1,e);
///         const double O13 = d(qx,qy,qz,2,e);
///         const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e);
///         const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e);
///         const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e);
///         const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e);
///         const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e);
///         const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e);
///         const double gX = u;
///         const double gY = v;
///         const double gZ = w;
///         QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ);
///         QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ);
///         QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);
///        }
///      }
///    }
///
///    for(int d=0; d<D1D; ++d){
///       for(int q=0,q<Q1D; ++q){
///         const int i = qi(q,d,Q1D);
///         const int j = dj(q,d,D1D);
///         const int k = qk(q,d,Q1D);
///         const int l = dl(q,d,D1D);
///         Bt[j][i] = b(q,d);
///         Gt[l][k] = g(q,d) * sign(q,d);
///      }
///     }
///
///     for(int qz=0;qz<Q1D;qz++){
///       for(int qy=0;qy<Q1D;++qy){
///          for(int dx=0; dx<D1D;++dx){
///            double u = 0.0, v = 0.0, w = 0.0;
///            for (int qx = 0; qx < Q1D; ++qx){
///              const int i = qi(qx,dx,Q1D);
///              const int j = dj(qx,dx,D1D);
///              const int k = qk(qx,dx,Q1D);
///              const int l = dl(qx,dx,D1D);
///              const double s = sign(qx,dx);
///              u += QQQ0[qz][qy][qx] * Gt[l][k] * s;
///              v += QQQ1[qz][qy][qx] * Bt[j][i];
///              w += QQQ2[qz][qy][qx] * Bt[j][i];
///            }
///            QQD0[qz][qy][dx] = u;
///            QQD1[qz][qy][dx] = v;
///            QQD2[qz][qy][dx] = w;
///          }
///       }
///     }
///
///     for(int qz=0;qz<Q1D;qz++){
///       for(int dy=0;dy<D1D;++dy){
///          for(int dx=0; dx<D1D;++dx){
///          double u = 0.0, v = 0.0, w = 0.0;
///          for (int qy = 0; qy < Q1D; ++qy){
///            const int i = qi(qy,dy,Q1D);
///            const int j = dj(qy,dy,D1D);
///            const int k = qk(qy,dy,Q1D);
///            const int l = dl(qy,dy,D1D);
///            const double s = sign(qy,dy);
///            u += QQD0[qz][qy][dx] * Bt[j][i];
///            v += QQD1[qz][qy][dx] * Gt[l][k] * s;
///            w += QQD2[qz][qy][dx] * Bt[j][i];
///           }
///          QDD0[qz][dy][dx] = u;
///          QDD1[qz][dy][dx] = v;
///          QDD2[qz][dy][dx] = w;
///        }
///      }
///    }
///
///    for(int dz=0;dz<D1D;dz++){
///      for(int dy=0;dy<D1D;++dy){
///        for(int dx=0; dx<D1D;++dx){
///           double u = 0.0, v = 0.0, w = 0.0;
///           for (int qz = 0; qz < Q1D; ++qz){
///              const int i = qi(qz,dz,Q1D);
///               const int j = dj(qz,dz,D1D);
///               const int k = qk(qz,dz,Q1D);
///               const int l = dl(qz,dz,D1D);
///               const double s = sign(qz,dz);
///               u += QDD0[qz][dy][dx] * Bt[j][i];
///               v += QDD1[qz][dy][dx] * Bt[j][i];
///               w += QDD2[qz][dy][dx] * Gt[l][k] * s;
///            }
///            y(dx,dy,dz,e) += (u + v + w);
///         }
///      }
///   }
///
/// } // element loop
///

#ifndef RAJAPerf_Apps_DIFFUSION3DPA_HPP
#define RAJAPerf_Apps_DIFFUSION3DPA_HPP

#define DIFFUSION3DPA_DATA_SETUP \
Real_ptr Basis = m_B; \
Real_ptr dBasis = m_G; \
Real_ptr D = m_D; \
Real_ptr X = m_X; \
Real_ptr Y = m_Y; \
Index_type NE = m_NE; \
const bool symmetric = true;

#include "common/KernelBase.hpp"
#include "FEM_MACROS.hpp"

#include "RAJA/RAJA.hpp"

//Number of Dofs/Qpts in 1D
#define DPA_D1D 3
#define DPA_Q1D 4
#define SYM 6
#define b(x, y) Basis[x + DPA_Q1D * y]
#define g(x, y) dBasis[x + DPA_Q1D * y]
#define dpaX_(dx, dy, dz, e)                                                      \
  X[dx + DPA_D1D * dy + DPA_D1D * DPA_D1D * dz + DPA_D1D * DPA_D1D * DPA_D1D * e]
#define dpaY_(dx, dy, dz, e)                                                      \
  Y[dx + DPA_D1D * dy + DPA_D1D * DPA_D1D * dz + DPA_D1D * DPA_D1D * DPA_D1D * e]
#define d(qx, qy, qz, s, e)                                                    \
  D[qx + DPA_Q1D * qy + DPA_Q1D * DPA_Q1D * qz + DPA_Q1D * DPA_Q1D * DPA_Q1D * s  +  DPA_Q1D * DPA_Q1D * DPA_Q1D * SYM * e]

// Half of B and G are stored in shared to get B, Bt, G and Gt.
// Indices computation for SmemPADiffusionApply3D.
static RAJA_HOST_DEVICE inline int qi(const int q, const int d, const int Q)
{
  return (q<=d) ? q : Q-1-q;
}

static RAJA_HOST_DEVICE inline int dj(const int q, const int d, const int D)
{
  return (q<=d) ? d : D-1-d;
}

static RAJA_HOST_DEVICE inline int qk(const int q, const int d, const int Q)
{
  return (q<=d) ? Q-1-q : q;
}

static RAJA_HOST_DEVICE inline int dl(const int q, const int d, const int D)
{
  return (q<=d) ? D-1-d : d;
}

static RAJA_HOST_DEVICE inline double sign(const int q, const int d)
{
  return (q<=d) ? -1.0 : 1.0;
}

#define DIFFUSION3DPA_0_GPU \
        constexpr int MQ1 = DPA_Q1D; \
        constexpr int MD1 = DPA_D1D; \
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1; \
        RAJA_TEAM_SHARED double sBG[MQ1*MD1]; \
        double (*B)[MD1] = (double (*)[MD1]) sBG; \
        double (*G)[MD1] = (double (*)[MD1]) sBG; \
        double (*Bt)[MQ1] = (double (*)[MQ1]) sBG; \
        double (*Gt)[MQ1] = (double (*)[MQ1]) sBG; \
        RAJA_TEAM_SHARED double sm0[3][MDQ*MDQ*MDQ]; \
        RAJA_TEAM_SHARED double sm1[3][MDQ*MDQ*MDQ]; \
        double (*s_X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2); \
        double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0); \
        double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1); \
        double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0); \
        double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1); \
        double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2); \
        double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0); \
        double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1); \
        double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2); \
        double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0); \
        double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1); \
        double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2); \
        double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0); \
        double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1); \
        double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);

#define DIFFUSION3DPA_0_CPU \
        constexpr int MQ1 = DPA_Q1D; \
        constexpr int MD1 = DPA_D1D; \
        constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1; \
        double sBG[MQ1*MD1]; \
        double (*B)[MD1] = (double (*)[MD1]) sBG; \
        double (*G)[MD1] = (double (*)[MD1]) sBG; \
        double (*Bt)[MQ1] = (double (*)[MQ1]) sBG; \
        double (*Gt)[MQ1] = (double (*)[MQ1]) sBG; \
        double sm0[3][MDQ*MDQ*MDQ]; \
        double sm1[3][MDQ*MDQ*MDQ]; \
        double (*s_X)[MD1][MD1]    = (double (*)[MD1][MD1]) (sm0+2); \
        double (*DDQ0)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+0); \
        double (*DDQ1)[MD1][MQ1] = (double (*)[MD1][MQ1]) (sm0+1); \
        double (*DQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+0); \
        double (*DQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+1); \
        double (*DQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm1+2); \
        double (*QQQ0)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+0); \
        double (*QQQ1)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+1); \
        double (*QQQ2)[MQ1][MQ1] = (double (*)[MQ1][MQ1]) (sm0+2); \
        double (*QQD0)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+0); \
        double (*QQD1)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+1); \
        double (*QQD2)[MQ1][MD1] = (double (*)[MQ1][MD1]) (sm1+2); \
        double (*QDD0)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+0); \
        double (*QDD1)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+1); \
        double (*QDD2)[MD1][MD1] = (double (*)[MD1][MD1]) (sm0+2);

#define DIFFUSION3DPA_1 \
        s_X[dz][dy][dx] = dpaX_(dx,dy,dz,e);

#define DIFFUSION3DPA_2 \
        const int i = qi(qx,dy,DPA_Q1D); \
        const int j = dj(qx,dy,DPA_D1D); \
        const int k = qk(qx,dy,DPA_Q1D); \
        const int l = dl(qx,dy,DPA_D1D); \
        B[i][j] = b(qx,dy); \
        G[k][l] = g(qx,dy) * sign(qx,dy); \

#define DIFFUSION3DPA_3 \
           double u = 0.0, v = 0.0; \
            RAJAPERF_UNROLL(MD1) \
            for (int dx = 0; dx < DPA_D1D; ++dx) \
            { \
               const int i = qi(qx,dx,DPA_Q1D); \
               const int j = dj(qx,dx,DPA_D1D); \
               const int k = qk(qx,dx,DPA_Q1D); \
               const int l = dl(qx,dx,DPA_D1D); \
               const double s = sign(qx,dx); \
               const double coords = s_X[dz][dy][dx]; \
               u += coords * B[i][j]; \
               v += coords * G[k][l] * s; \
             } \
             DDQ0[dz][dy][qx] = u; \
             DDQ1[dz][dy][qx] = v;

#define DIFFUSION3DPA_4 \
   double u = 0.0, v = 0.0, w = 0.0; \
   RAJAPERF_UNROLL(MD1)  \
   for (int dy = 0; dy < DPA_D1D; ++dy) \
   { \
      const int i = qi(qy,dy,DPA_Q1D); \
      const int j = dj(qy,dy,DPA_D1D); \
      const int k = qk(qy,dy,DPA_Q1D); \
      const int l = dl(qy,dy,DPA_D1D); \
      const double s = sign(qy,dy); \
      u += DDQ1[dz][dy][qx] * B[i][j]; \
      v += DDQ0[dz][dy][qx] * G[k][l] * s; \
      w += DDQ0[dz][dy][qx] * B[i][j]; \
   } \
   DQQ0[dz][qy][qx] = u; \
   DQQ1[dz][qy][qx] = v; \
   DQQ2[dz][qy][qx] = w;

#define DIFFUSION3DPA_5 \
               double u = 0.0, v = 0.0, w = 0.0; \
               RAJAPERF_UNROLL(MD1) \
               for (int dz = 0; dz < DPA_D1D; ++dz) \
               { \
                  const int i = qi(qz,dz,DPA_Q1D); \
                  const int j = dj(qz,dz,DPA_D1D); \
                  const int k = qk(qz,dz,DPA_Q1D); \
                  const int l = dl(qz,dz,DPA_D1D); \
                  const double s = sign(qz,dz); \
                  u += DQQ0[dz][qy][qx] * B[i][j]; \
                  v += DQQ1[dz][qy][qx] * B[i][j]; \
                  w += DQQ2[dz][qy][qx] * G[k][l] * s; \
               } \
               const double O11 = d(qx,qy,qz,0,e); \
               const double O12 = d(qx,qy,qz,1,e); \
               const double O13 = d(qx,qy,qz,2,e); \
               const double O21 = symmetric ? O12 : d(qx,qy,qz,3,e); \
               const double O22 = symmetric ? d(qx,qy,qz,3,e) : d(qx,qy,qz,4,e); \
               const double O23 = symmetric ? d(qx,qy,qz,4,e) : d(qx,qy,qz,5,e); \
               const double O31 = symmetric ? O13 : d(qx,qy,qz,6,e); \
               const double O32 = symmetric ? O23 : d(qx,qy,qz,7,e); \
               const double O33 = symmetric ? d(qx,qy,qz,5,e) : d(qx,qy,qz,8,e); \
               const double gX = u; \
               const double gY = v; \
               const double gZ = w; \
               QQQ0[qz][qy][qx] = (O11*gX) + (O12*gY) + (O13*gZ); \
               QQQ1[qz][qy][qx] = (O21*gX) + (O22*gY) + (O23*gZ); \
               QQQ2[qz][qy][qx] = (O31*gX) + (O32*gY) + (O33*gZ);

#define DIFFUSION3DPA_6 \
               const int i = qi(q,d,DPA_Q1D); \
               const int j = dj(q,d,DPA_D1D); \
               const int k = qk(q,d,DPA_Q1D); \
               const int l = dl(q,d,DPA_D1D); \
               Bt[j][i] = b(q,d); \
               Gt[l][k] = g(q,d) * sign(q,d);

#define DIFFUSION3DPA_7 \
            double u = 0.0, v = 0.0, w = 0.0; \
            RAJAPERF_UNROLL(MQ1) \
            for (int qx = 0; qx < DPA_Q1D; ++qx) \
            { \
              const int i = qi(qx,dx,DPA_Q1D); \
              const int j = dj(qx,dx,DPA_D1D); \
              const int k = qk(qx,dx,DPA_Q1D); \
              const int l = dl(qx,dx,DPA_D1D); \
              const double s = sign(qx,dx); \
              u += QQQ0[qz][qy][qx] * Gt[l][k] * s; \
              v += QQQ1[qz][qy][qx] * Bt[j][i]; \
              w += QQQ2[qz][qy][qx] * Bt[j][i]; \
            } \
            QQD0[qz][qy][dx] = u; \
            QQD1[qz][qy][dx] = v; \
            QQD2[qz][qy][dx] = w;

#define DIFFUSION3DPA_8 \
        double u = 0.0, v = 0.0, w = 0.0; \
        RAJAPERF_UNROLL(DPA_Q1D)  \
        for (int qy = 0; qy < DPA_Q1D; ++qy) \
        { \
          const int i = qi(qy,dy,DPA_Q1D); \
          const int j = dj(qy,dy,DPA_D1D); \
          const int k = qk(qy,dy,DPA_Q1D); \
          const int l = dl(qy,dy,DPA_D1D); \
          const double s = sign(qy,dy); \
          u += QQD0[qz][qy][dx] * Bt[j][i]; \
          v += QQD1[qz][qy][dx] * Gt[l][k] * s; \
          w += QQD2[qz][qy][dx] * Bt[j][i]; \
        } \
        QDD0[qz][dy][dx] = u; \
        QDD1[qz][dy][dx] = v; \
        QDD2[qz][dy][dx] = w;

#define DIFFUSION3DPA_9 \
        double u = 0.0, v = 0.0, w = 0.0; \
        RAJAPERF_UNROLL(MQ1) \
        for (int qz = 0; qz < DPA_Q1D; ++qz)  \
        {                                     \
          const int i = qi(qz,dz,DPA_Q1D); \
          const int j = dj(qz,dz,DPA_D1D); \
          const int k = qk(qz,dz,DPA_Q1D); \
          const int l = dl(qz,dz,DPA_D1D); \
          const double s = sign(qz,dz);    \
          u += QDD0[qz][dy][dx] * Bt[j][i];     \
          v += QDD1[qz][dy][dx] * Bt[j][i];     \
          w += QDD2[qz][dy][dx] * Gt[l][k] * s; \
        }                                       \
        dpaY_(dx,dy,dz,e) += (u + v + w);

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
  static const size_t default_gpu_block_size = DPA_Q1D * DPA_Q1D * DPA_Q1D;
  using gpu_block_sizes_type = gpu_block_size::list_type<default_gpu_block_size>;

  Real_ptr m_B;
  Real_ptr m_Bt;
  Real_ptr m_G;
  Real_ptr m_Gt;
  Real_ptr m_D;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_NE;
  Index_type m_NE_default;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
