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
/// https://github.com/mfem/mfem/blob/master/fem/bilininteg_convection_pa.cpp
///
///
/// for(int e = 0; e < NE; ++e) {
///
///   constexpr int max_D1D = CPA_D1D;
///   constexpr int max_Q1D = CPA_Q1D;
///   constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
///   MFEM_SHARED double sm0[max_DQ*max_DQ*max_DQ];
///   MFEM_SHARED double sm1[max_DQ*max_DQ*max_DQ];
///   MFEM_SHARED double sm2[max_DQ*max_DQ*max_DQ];
///   MFEM_SHARED double sm3[max_DQ*max_DQ*max_DQ];
///   MFEM_SHARED double sm4[max_DQ*max_DQ*max_DQ];
///   MFEM_SHARED double sm5[max_DQ*max_DQ*max_DQ];
///
///   double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
///   for(int dz = 0; dz < CPA_D1D; ++dz)
///   {
///     for(int dy = 0; dy < CPA_D1D; ++dy)
///     {
///       for(int dx = 0; dx < CPA_D1D; ++dx)
///       {
///         u[dz][dy][dx] = cpaX_(dx,dy,dz,e);
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
///   double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
///   for(int dz = 0; dz < CPA_D1D; ++dz)
///   {
///     for(int dy = 0; dy < CPA_D1D; ++dy)
///     {
///       for(int qx = 0; qx < CPA_Q1D; ++qx)
///       {
///         double Bu_ = 0.0;
///         double Gu_ = 0.0;
///         for(int dx = 0; dx < CPA_D1D; ++dx)
///         {
///           const double bx = cpa_B(qx,dx);
///           const double gx = cpa_G(qx,dx);
///           const double x = u[dz][dy][dx];
///           Bu_ += bx * x;
///           Gu_ += gx * x;
///         }
///         Bu[dz][dy][qx] = Bu_;
///         Gu[dz][dy][qx] = Gu_;
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
///   double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
///   double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
///   for(int dz = 0; dz < CPA_D1D; ++dz)
///   {
///     for(int qx = 0; qx < CPA_Q1D; ++qx)
///     {
///       for(int qy = 0; qy < CPA_Q1D; ++qy)
///       {
///         double BBu_ = 0.0;
///         double GBu_ = 0.0;
///         double BGu_ = 0.0;
///         for(int dy = 0; dy < CPA_D1D; ++dy)
///         {
///           const double bx = cpa_B(qy,dy);
///           const double gx = cpa_G(qy,dy);
///           BBu_ += bx * Bu[dz][dy][qx];
///           GBu_ += gx * Bu[dz][dy][qx];
///           BGu_ += bx * Gu[dz][dy][qx];
///         }
///         BBu[dz][qy][qx] = BBu_;
///         GBu[dz][qy][qx] = GBu_;
///         BGu[dz][qy][qx] = BGu_;
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
///   double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
///   double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
///   for(int qx = 0; qx < CPA_Q1D; ++qx)
///   {
///     for(int qy = 0; qy < CPA_Q1D; ++qy)
///     {
///       for(int qz = 0; qz < CPA_Q1D; ++qz)
///       {
///         double GBBu_ = 0.0;
///         double BGBu_ = 0.0;
///         double BBGu_ = 0.0;
///         for(int dz = 0; dz < CPA_D1D; ++dz)
///         {
///           const double bx = cpa_B(qz,dz);
///           const double gx = cpa_G(qz,dz);
///           GBBu_ += gx * BBu[dz][qy][qx];
///           BGBu_ += bx * GBu[dz][qy][qx];
///           BBGu_ += bx * BGu[dz][qy][qx];
///         }
///         GBBu[qz][qy][qx] = GBBu_;
///         BGBu[qz][qy][qx] = BGBu_;
///         BBGu[qz][qy][qx] = BBGu_;
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
///   for(int qz = 0; qz < CPA_Q1D; ++qz)
///   {
///     for(int qy = 0; qy < CPA_Q1D; ++qy)
///     {
///       for(int qx = 0; qx < CPA_Q1D; ++qx)
///       {
///         const double O1 = cpa_op(qx,qy,qz,0,e);
///         const double O2 = cpa_op(qx,qy,qz,1,e);
///         const double O3 = cpa_op(qx,qy,qz,2,e);
///
///         const double gradX = BBGu[qz][qy][qx];
///         const double gradY = BGBu[qz][qy][qx];
///         const double gradZ = GBBu[qz][qy][qx];
///
///         DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
///   for(int qx = 0; qx < CPA_Q1D; ++qx)
///   {
///     for(int qy = 0; qy < CPA_Q1D; ++qy)
///     {
///       for(int dz = 0; dz < CPA_D1D; ++dz)
///       {
///          double BDGu_ = 0.0;
///          for(int qz = 0; qz < CPA_Q1D; ++qz)
///          {
///             const double w = cpa_Bt(dz,qz);
///             BDGu_ += w * DGu[qz][qy][qx];
///          }
///          BDGu[dz][qy][qx] = BDGu_;
///       }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;
///   for(int dz = 0; dz < CPA_D1D; ++dz)
///   {
///     for(int qx = 0; qx < CPA_Q1D; ++qx)
///      {
///        for(int dy = 0; dy < CPA_D1D; ++dy)
///         {
///            double BBDGu_ = 0.0;
///            for(int qy = 0; qy < CPA_Q1D; ++qy)
///            {
///              const double w = cpa_Bt(dy,qy);
///              BBDGu_ += w * BDGu[dz][qy][qx];
///           }
///           BBDGu[dz][dy][qx] = BBDGu_;
///        }
///     }
///   }
///   MFEM_SYNC_THREAD;
///   for(int dz = 0; dz < CPA_D1D; ++dz)
///   {
///     for(int dy = 0; dy < CPA_D1D; ++dy)
///     {
///       for(int dx = 0; dx < CPA_D1D; ++dx)
///       {
///         double BBBDGu = 0.0;
///         for(int qx = 0; qx < CPA_Q1D; ++qx)
///         {
///           const double w = cpa_Bt(dx,qx);
///           BBBDGu += w * BBDGu[dz][dy][qx];
///         }
///         cpaY_(dx,dy,dz,e) += BBBDGu;
///       }
///     }
///   }
/// } // element loop
///

#ifndef RAJAPerf_Apps_CONVECTION3DPA_HPP
#define RAJAPerf_Apps_CONVECTION3DPA_HPP

#define CONVECTION3DPA_DATA_SETUP \
Real_ptr Basis = m_B; \
Real_ptr tBasis = m_Bt; \
Real_ptr dBasis = m_G; \
Real_ptr D = m_D; \
Real_ptr X = m_X; \
Real_ptr Y = m_Y; \
Index_type NE = m_NE;

#include "common/KernelBase.hpp"
#include "FEM_MACROS.hpp"

#include "RAJA/RAJA.hpp"

//Number of Dofs/Qpts in 1D
#define CPA_D1D 3
#define CPA_Q1D 4
#define CPA_VDIM 3
#define cpa_B(x, y) Basis[x + CPA_Q1D * y]
#define cpa_Bt(x, y) tBasis[x + CPA_D1D * y]
#define cpa_G(x, y) dBasis[x + CPA_Q1D * y]
#define cpaX_(dx, dy, dz, e)                                                     \
  X[dx + CPA_D1D * dy + CPA_D1D * CPA_D1D * dz + CPA_D1D * CPA_D1D * CPA_D1D * e]
#define cpaY_(dx, dy, dz, e)                                                      \
  Y[dx + CPA_D1D * dy + CPA_D1D * CPA_D1D * dz + CPA_D1D * CPA_D1D * CPA_D1D * e]
#define cpa_op(qx, qy, qz, d, e)                                       \
  D[qx + CPA_Q1D * qy + CPA_Q1D * CPA_Q1D * qz + CPA_Q1D * CPA_Q1D * CPA_Q1D * d  +  CPA_VDIM * CPA_Q1D * CPA_Q1D * CPA_Q1D * e]

#define CONVECTION3DPA_0_GPU \
  constexpr int max_D1D = CPA_D1D; \
  constexpr int max_Q1D = CPA_Q1D; \
  constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D; \
  RAJA_TEAM_SHARED double sm0[max_DQ*max_DQ*max_DQ]; \
  RAJA_TEAM_SHARED double sm1[max_DQ*max_DQ*max_DQ]; \
  RAJA_TEAM_SHARED double sm2[max_DQ*max_DQ*max_DQ]; \
  RAJA_TEAM_SHARED double sm3[max_DQ*max_DQ*max_DQ]; \
  RAJA_TEAM_SHARED double sm4[max_DQ*max_DQ*max_DQ]; \
  RAJA_TEAM_SHARED double sm5[max_DQ*max_DQ*max_DQ]; \
  double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0; \
  double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1; \
  double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2; \
  double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3; \
  double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4; \
  double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5; \
  double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0; \
  double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1; \
  double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2; \
  double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;  \
  double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4; \
  double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;


#define CONVECTION3DPA_0_CPU \
  constexpr int max_D1D = CPA_D1D; \
  constexpr int max_Q1D = CPA_Q1D; \
  constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D; \
  double sm0[max_DQ*max_DQ*max_DQ]; \
  double sm1[max_DQ*max_DQ*max_DQ]; \
  double sm2[max_DQ*max_DQ*max_DQ]; \
  double sm3[max_DQ*max_DQ*max_DQ]; \
  double sm4[max_DQ*max_DQ*max_DQ]; \
  double sm5[max_DQ*max_DQ*max_DQ]; \
  double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0; \
  double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1; \
  double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2; \
  double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3; \
  double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4; \
  double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5; \
  double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0; \
  double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1; \
  double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2; \
  double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;  \
  double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4; \
  double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;

#define CONVECTION3DPA_1 \
  u[dz][dy][dx] = cpaX_(dx,dy,dz,e);

#define CONVECTION3DPA_2 \
  double Bu_ = 0.0; \
  double Gu_ = 0.0; \
  for (int dx = 0; dx < CPA_D1D; ++dx) \
  { \
    const double bx = cpa_B(qx,dx); \
    const double gx = cpa_G(qx,dx); \
    const double x = u[dz][dy][dx]; \
    Bu_ += bx * x; \
    Gu_ += gx * x; \
  } \
  Bu[dz][dy][qx] = Bu_; \
  Gu[dz][dy][qx] = Gu_;

#define CONVECTION3DPA_3 \
  double BBu_ = 0.0; \
  double GBu_ = 0.0; \
  double BGu_ = 0.0; \
  for (int dy = 0; dy < CPA_D1D; ++dy) \
  { \
    const double bx = cpa_B(qy,dy); \
    const double gx = cpa_G(qy,dy); \
    BBu_ += bx * Bu[dz][dy][qx]; \
    GBu_ += gx * Bu[dz][dy][qx]; \
    BGu_ += bx * Gu[dz][dy][qx]; \
  } \
  BBu[dz][qy][qx] = BBu_; \
  GBu[dz][qy][qx] = GBu_; \
  BGu[dz][qy][qx] = BGu_;

#define CONVECTION3DPA_4 \
  double GBBu_ = 0.0; \
  double BGBu_ = 0.0; \
  double BBGu_ = 0.0; \
  for (int dz = 0; dz < CPA_D1D; ++dz) \
  { \
    const double bx = cpa_B(qz,dz); \
    const double gx = cpa_G(qz,dz); \
    GBBu_ += gx * BBu[dz][qy][qx]; \
    BGBu_ += bx * GBu[dz][qy][qx]; \
    BBGu_ += bx * BGu[dz][qy][qx]; \
  } \
  GBBu[qz][qy][qx] = GBBu_; \
  BGBu[qz][qy][qx] = BGBu_; \
  BBGu[qz][qy][qx] = BBGu_;

#define CONVECTION3DPA_5 \
  const double O1 = cpa_op(qx,qy,qz,0,e); \
  const double O2 = cpa_op(qx,qy,qz,1,e); \
  const double O3 = cpa_op(qx,qy,qz,2,e); \
  const double gradX = BBGu[qz][qy][qx]; \
  const double gradY = BGBu[qz][qy][qx]; \
  const double gradZ = GBBu[qz][qy][qx]; \
  DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);

#define CONVECTION3DPA_6 \
  double BDGu_ = 0.0; \
  for (int qz = 0; qz < CPA_Q1D; ++qz) \
  { \
    const double w = cpa_Bt(dz,qz); \
    BDGu_ += w * DGu[qz][qy][qx]; \
   } \
   BDGu[dz][qy][qx] = BDGu_;

#define CONVECTION3DPA_7 \
  double BBDGu_ = 0.0; \
  for (int qy = 0; qy < CPA_Q1D; ++qy) \
  { \
    const double w = cpa_Bt(dy,qy); \
    BBDGu_ += w * BDGu[dz][qy][qx]; \
  } \
  BBDGu[dz][dy][qx] = BBDGu_; \

#define CONVECTION3DPA_8 \
  double BBBDGu = 0.0; \
  for (int qx = 0; qx < CPA_Q1D; ++qx) \
  { \
    const double w = cpa_Bt(dx,qx); \
    BBBDGu += w * BBDGu[dz][dy][qx]; \
  } \
  cpaY_(dx,dy,dz,e) += BBBDGu;

namespace rajaperf
{
class RunParams;

namespace apps
{

class CONVECTION3DPA : public KernelBase
{
public:

  CONVECTION3DPA(const RunParams& params);

  ~CONVECTION3DPA();

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
  static const size_t default_gpu_block_size = CPA_Q1D * CPA_Q1D * CPA_Q1D;
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
