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
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "DEL_DOT_VEC_2D.hxx"

#include "AppsData.hxx"
#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  typedef RAJA::Real_type* __restrict__ UnalignedReal_ptr;

#define DEL_DOT_VEC_2D_DATA \
  RAJA::Real_ptr x = m_x; \
  RAJA::Real_ptr y = m_y; \
  RAJA::Real_ptr xdot = m_xdot; \
  RAJA::Real_ptr ydot = m_ydot; \
  RAJA::Real_ptr div = m_div; \
  UnalignedReal_ptr x1,x2,x3,x4 ; \
  UnalignedReal_ptr y1,y2,y3,y4 ; \
  UnalignedReal_ptr fx1,fx2,fx3,fx4 ; \
  UnalignedReal_ptr fy1,fy2,fy3,fy4 ; \
  const RAJA::Real_type ptiny = m_ptiny; \
  const RAJA::Real_type half = m_half;


#define DEL_DOT_VEC_2D_INDEX_BODY \
  RAJA::Index_type i  = m_domain->real_zones[ii] ;

#define DEL_DOT_VEC_2D_BODY \
  RAJA::Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ; \
  RAJA::Real_type xj  = half * ( x2[i]  + x3[i]  - x4[i]  - x1[i]  ) ; \
 \
  RAJA::Real_type yi  = half * ( y1[i]  + y2[i]  - y3[i]  - y4[i]  ) ; \
  RAJA::Real_type yj  = half * ( y2[i]  + y3[i]  - y4[i]  - y1[i]  ) ; \
 \
  RAJA::Real_type fxi = half * ( fx1[i] + fx2[i] - fx3[i] - fx4[i] ) ; \
  RAJA::Real_type fxj = half * ( fx2[i] + fx3[i] - fx4[i] - fx1[i] ) ; \
 \
  RAJA::Real_type fyi = half * ( fy1[i] + fy2[i] - fy3[i] - fy4[i] ) ; \
  RAJA::Real_type fyj = half * ( fy2[i] + fy3[i] - fy4[i] - fy1[i] ) ; \
 \
  RAJA::Real_type rarea  = 1.0 / ( xi * yj - xj * yi + ptiny ) ; \
 \
  RAJA::Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ; \
 \
  RAJA::Real_type dfydy  = rarea * ( fyj * xi - fyi * xj ) ; \
 \
  RAJA::Real_type affine = ( fy1[i] + fy2[i] + fy3[i] + fy4[i] ) / \
                     ( y1[i]  + y2[i]  + y3[i]  + y4[i]  ) ; \
 \
  div[i] = dfxdx + dfydy + affine ;


DEL_DOT_VEC_2D::DEL_DOT_VEC_2D(const RunParams& params)
  : KernelBase(rajaperf::Apps_DEL_DOT_VEC_2D, params)
{
  setDefaultSize(312);  // See rzmax in ADomain struct
  setDefaultSamples(2000);

  m_domain = new ADomain(getRunSize(), /* ndims = */ 2);
}

DEL_DOT_VEC_2D::~DEL_DOT_VEC_2D() 
{
  delete m_domain;
}

void DEL_DOT_VEC_2D::setUp(VariantID vid)
{
  int max_loop_index = m_domain->lrn;

  allocAndInitAligned(m_x, max_loop_index, vid);
  allocAndInitAligned(m_y, max_loop_index, vid);
  allocAndInitAligned(m_xdot, max_loop_index, vid);
  allocAndInitAligned(m_ydot, max_loop_index, vid);
  allocAndInitAligned(m_div, max_loop_index, vid);

  m_ptiny = 1.0e-20;
  m_half = 0.5;
}

void DEL_DOT_VEC_2D::runKernel(VariantID vid)
{
  int run_samples = getRunSamples();
  RAJA::Index_type lbegin = 0;
  RAJA::Index_type lend = m_domain->n_real_zones;

  switch ( vid ) {

    case Baseline_Seq : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D((*m_domain), x,x1,x2,x3,x4) ;
      NDSET2D((*m_domain), y,y1,y2,y3,y4) ;
      NDSET2D((*m_domain), xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D((*m_domain), ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (RAJA::Index_type ii = lbegin ; ii < lend ; ++ii ) {
          DEL_DOT_VEC_2D_INDEX_BODY;
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      DEL_DOT_VEC_2D_DATA;

      NDSET2D((*m_domain), x,x1,x2,x3,x4) ;
      NDSET2D((*m_domain), y,y1,y2,y3,y4) ;
      NDSET2D((*m_domain), xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D((*m_domain), ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(m_domain->real_zones, lend, [=](int i) {
          DEL_DOT_VEC_2D_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

    case Baseline_OpenMP : {
#if defined(_OPENMP)      
      DEL_DOT_VEC_2D_DATA;

      NDSET2D((*m_domain), x,x1,x2,x3,x4) ;
      NDSET2D((*m_domain), y,y1,y2,y3,y4) ;
      NDSET2D((*m_domain), xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D((*m_domain), ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for 
        for (RAJA::Index_type ii = lbegin ; ii < lend ; ++ii ) {
          DEL_DOT_VEC_2D_INDEX_BODY;
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();
#endif
      break;
    }

    case RAJALike_OpenMP : {
      // Not applicable
      break;
    }

    case RAJA_OpenMP : {
#if defined(_OPENMP)      

      DEL_DOT_VEC_2D_DATA;

      NDSET2D((*m_domain), x,x1,x2,x3,x4) ;
      NDSET2D((*m_domain), y,y1,y2,y3,y4) ;
      NDSET2D((*m_domain), xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D((*m_domain), ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(m_domain->real_zones, lend, 
        [=](int i) {
          DEL_DOT_VEC_2D_BODY;
        });

      }
      stopTimer();
#endif
      break;
    }

    case Baseline_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
      break;
    }

#if 0
    case Baseline_OpenMP4x :
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
  freeAligned(m_x);
  freeAligned(m_y);
  freeAligned(m_xdot);
  freeAligned(m_ydot);
  freeAligned(m_div);
  
  if (vid == Baseline_CUDA || vid == RAJA_CUDA) {
    // De-allocate device memory here.
  }
}

} // end namespace apps
} // end namespace rajaperf
