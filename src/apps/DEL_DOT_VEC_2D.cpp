/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel DEL_DOT_VEC_2D.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "DEL_DOT_VEC_2D.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define DEL_DOT_VEC_2D_DATA \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr xdot = m_xdot; \
  ResReal_ptr ydot = m_ydot; \
  ResReal_ptr div = m_div; \
  Index_ptr real_zones = m_domain->real_zones; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  ResReal_ptr x1,x2,x3,x4 ; \
  ResReal_ptr y1,y2,y3,y4 ; \
  ResReal_ptr fx1,fx2,fx3,fx4 ; \
  ResReal_ptr fy1,fy2,fy3,fy4 ;


#define DEL_DOT_VEC_2D_BODY \
  Index_type i = real_zones[ii]; \
\
  Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ; \
  Real_type xj  = half * ( x2[i]  + x3[i]  - x4[i]  - x1[i]  ) ; \
 \
  Real_type yi  = half * ( y1[i]  + y2[i]  - y3[i]  - y4[i]  ) ; \
  Real_type yj  = half * ( y2[i]  + y3[i]  - y4[i]  - y1[i]  ) ; \
 \
  Real_type fxi = half * ( fx1[i] + fx2[i] - fx3[i] - fx4[i] ) ; \
  Real_type fxj = half * ( fx2[i] + fx3[i] - fx4[i] - fx1[i] ) ; \
 \
  Real_type fyi = half * ( fy1[i] + fy2[i] - fy3[i] - fy4[i] ) ; \
  Real_type fyj = half * ( fy2[i] + fy3[i] - fy4[i] - fy1[i] ) ; \
 \
  Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ; \
 \
  Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ; \
 \
  Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ; \
 \
  Real_type affine = ( fy1[i] + fy2[i] + fy3[i] + fy4[i] ) / \
                     ( y1[i]  + y2[i]  + y3[i]  + y4[i]  ) ; \
 \
  div[i] = dfxdx + dfydy + affine ;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define DEL_DOT_VEC_2D_DATA_SETUP_CUDA \
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr xdot; \
  Real_ptr ydot; \
  Real_ptr div; \
  Index_ptr real_zones; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  Real_ptr x1,x2,x3,x4 ; \
  Real_ptr y1,y2,y3,y4 ; \
  Real_ptr fx1,fx2,fx3,fx4 ; \
  Real_ptr fy1,fy2,fy3,fy4 ; \
\
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend); \
  allocAndInitCudaDeviceData(xdot, m_xdot, iend); \
  allocAndInitCudaDeviceData(ydot, m_ydot, iend); \
  allocAndInitCudaDeviceData(div, m_div, iend); \
  allocAndInitCudaDeviceData(real_zones, m_domain->real_zones, m_domain->n_real_zones);

#define DEL_DOT_VEC_2D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_div, div, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(xdot); \
  deallocCudaDeviceData(ydot); \
  deallocCudaDeviceData(div);

__global__ void deldotvec2d(Real_ptr div, 
                            const Real_ptr x1, const Real_ptr x2,
                            const Real_ptr x3, const Real_ptr x4,
                            const Real_ptr y1, const Real_ptr y2,
                            const Real_ptr y3, const Real_ptr y4,
                            const Real_ptr fx1, const Real_ptr fx2,
                            const Real_ptr fx3, const Real_ptr fx4,
                            const Real_ptr fy1, const Real_ptr fy2,
                            const Real_ptr fy3, const Real_ptr fy4,
                            const Index_ptr real_zones,
                            const Real_type half, const Real_type ptiny,
                            Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   if (ii < iend) {
     DEL_DOT_VEC_2D_BODY;
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


DEL_DOT_VEC_2D::DEL_DOT_VEC_2D(const RunParams& params)
  : KernelBase(rajaperf::Apps_DEL_DOT_VEC_2D, params)
{
  setDefaultSize(312);  // See rzmax in ADomain struct
  setDefaultReps(1200);

  m_domain = new ADomain(getRunSize(), /* ndims = */ 2);
}

DEL_DOT_VEC_2D::~DEL_DOT_VEC_2D() 
{
  delete m_domain;
}

Index_type DEL_DOT_VEC_2D::getItsPerRep() const 
{ 
  return m_domain->n_real_zones;
}

void DEL_DOT_VEC_2D::setUp(VariantID vid)
{
  int max_loop_index = m_domain->lrn;

  allocAndInitData(m_x, max_loop_index, vid);
  allocAndInitData(m_y, max_loop_index, vid);
  allocAndInitData(m_xdot, max_loop_index, vid);
  allocAndInitData(m_ydot, max_loop_index, vid);
  allocAndInitData(m_div, max_loop_index, vid);

  m_ptiny = 1.0e-20;
  m_half = 0.5;
}

void DEL_DOT_VEC_2D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  switch ( vid ) {

    case Base_Seq : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int ii) {
          DEL_DOT_VEC_2D_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(_OPENMP)      
    case Base_OpenMP : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for 
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // Not applicable
      break;
    }

    case RAJA_OpenMP : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int ii) {
          DEL_DOT_VEC_2D_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      DEL_DOT_VEC_2D_DATA_SETUP_CUDA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

        deldotvec2d<<<grid_size, block_size>>>(div, 
                                               x1, x2, x3, x4,
                                               y1, y2, y3, y4,
                                               fx1, fx2, fx3, fx4,
                                               fy1, fy2, fy3, fy4,
                                               real_zones,
                                               half, ptiny,
                                               iend);

      }
      stopTimer();

      DEL_DOT_VEC_2D_DATA_TEARDOWN_CUDA;

      break;
    }

    case RAJA_CUDA : {

      DEL_DOT_VEC_2D_DATA_SETUP_CUDA;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type ii) {
           DEL_DOT_VEC_2D_BODY;
         });

      }
      stopTimer();

      DEL_DOT_VEC_2D_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

#if 0
    case Base_OpenMP4x :
    case RAJA_OpenMP4x : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }
}

void DEL_DOT_VEC_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_div, getRunSize());
}

void DEL_DOT_VEC_2D::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_xdot);
  deallocData(m_ydot);
  deallocData(m_div);
}

} // end namespace apps
} // end namespace rajaperf
