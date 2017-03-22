/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel VOL3D_CALC.
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


#include "VOL3D_CALC.hxx"

#include "AppsData.hxx"
#include "common/DataUtils.hxx"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  typedef RAJA::Real_type* __restrict__ UnalignedReal_ptr;

#define VOL3D_CALC_DATA \
  RAJA::Real_ptr x = m_x; \
  RAJA::Real_ptr y = m_y; \
  RAJA::Real_ptr z = m_z; \
  RAJA::Real_ptr vol = m_vol; \
  UnalignedReal_ptr x0,x1,x2,x3,x4,x5,x6,x7 ; \
  UnalignedReal_ptr y0,y1,y2,y3,y4,y5,y6,y7 ; \
  UnalignedReal_ptr z0,z1,z2,z3,z4,z5,z6,z7 ;  \
  const RAJA::Real_type vnormq = m_vnormq; \
  ADomain domain(run_size, /* ndims = */ 3);

#define VOL3D_CALC_BODY(i) \
  RAJA::Real_type x71 = x7[i] - x1[i] ; \
  RAJA::Real_type x72 = x7[i] - x2[i] ; \
  RAJA::Real_type x74 = x7[i] - x4[i] ; \
  RAJA::Real_type x30 = x3[i] - x0[i] ; \
  RAJA::Real_type x50 = x5[i] - x0[i] ; \
  RAJA::Real_type x60 = x6[i] - x0[i] ; \
 \
  RAJA::Real_type y71 = y7[i] - y1[i] ; \
  RAJA::Real_type y72 = y7[i] - y2[i] ; \
  RAJA::Real_type y74 = y7[i] - y4[i] ; \
  RAJA::Real_type y30 = y3[i] - y0[i] ; \
  RAJA::Real_type y50 = y5[i] - y0[i] ; \
  RAJA::Real_type y60 = y6[i] - y0[i] ; \
 \
  RAJA::Real_type z71 = z7[i] - z1[i] ; \
  RAJA::Real_type z72 = z7[i] - z2[i] ; \
  RAJA::Real_type z74 = z7[i] - z4[i] ; \
  RAJA::Real_type z30 = z3[i] - z0[i] ; \
  RAJA::Real_type z50 = z5[i] - z0[i] ; \
  RAJA::Real_type z60 = z6[i] - z0[i] ; \
 \
  RAJA::Real_type xps = x71 + x60 ; \
  RAJA::Real_type yps = y71 + y60 ; \
  RAJA::Real_type zps = z71 + z60 ; \
 \
  RAJA::Real_type cyz = y72 * z30 - z72 * y30 ; \
  RAJA::Real_type czx = z72 * x30 - x72 * z30 ; \
  RAJA::Real_type cxy = x72 * y30 - y72 * x30 ; \
  vol[i] = xps * cyz + yps * czx + zps * cxy ; \
 \
  xps = x72 + x50 ; \
  yps = y72 + y50 ; \
  zps = z72 + z50 ; \
 \
  cyz = y74 * z60 - z74 * y60 ; \
  czx = z74 * x60 - x74 * z60 ; \
  cxy = x74 * y60 - y74 * x60 ; \
  vol[i] += xps * cyz + yps * czx + zps * cxy ; \
 \
  xps = x74 + x30 ; \
  yps = y74 + y30 ; \
  zps = z74 + z30 ; \
 \
  cyz = y71 * z50 - z71 * y50 ; \
  czx = z71 * x50 - x71 * z50 ; \
  cxy = x71 * y50 - y71 * x50 ; \
  vol[i] += xps * cyz + yps * czx + zps * cxy ; \
 \
  vol[i] *= vnormq ;


VOL3D_CALC::VOL3D_CALC(const RunParams& params)
  : KernelBase(rajaperf::Apps_VOL3D_CALC, params)
{
  setDefaultSize(256);
  setDefaultSamples(3200);
}

VOL3D_CALC::~VOL3D_CALC() 
{
}

void VOL3D_CALC::setUp(VariantID vid)
{
  allocAndInitAligned(m_x, getRunSize(), vid);
  allocAndInitAligned(m_y, getRunSize(), vid);
  allocAndInitAligned(m_x, getRunSize(), vid);
  allocAndInitAligned(m_vol, getRunSize(), vid);

  m_vnormq = 0.083333333333333333; /* vnormq = 1/12 */  
}

void VOL3D_CALC::runKernel(VariantID vid)
{
  int run_size = getRunSize();
  int run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      VOL3D_CALC_DATA;

      NDPTRSET(x,x0,x1,x2,x3,x4,x5,x6,x7) ;
      NDPTRSET(y,y0,y1,y2,y3,y4,y5,y6,y7) ;
      NDPTRSET(z,z0,z1,z2,z3,z4,z5,z6,z7) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (RAJA::Index_type i = domain.fpz ; i <= domain.lpz ; i++ ) {
          VOL3D_CALC_BODY(i);
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      VOL3D_CALC_DATA;

      NDPTRSET(x,x0,x1,x2,x3,x4,x5,x6,x7) ;
      NDPTRSET(y,y0,y1,y2,y3,y4,y5,y6,y7) ;
      NDPTRSET(z,z0,z1,z2,z3,z4,z5,z6,z7) ;
 
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(domain.fpz, domain.lpz + 1, [=](int i) {
          VOL3D_CALC_BODY(i);
        }); 

      }
      stopTimer(); 

      break;
    }

    case Baseline_OpenMP : {
#if defined(_OPENMP)      
      VOL3D_CALC_DATA;

      NDPTRSET(x,x0,x1,x2,x3,x4,x5,x6,x7) ;
      NDPTRSET(y,y0,y1,y2,y3,y4,y5,y6,y7) ;
      NDPTRSET(z,z0,z1,z2,z3,z4,z5,z6,z7) ;
 
      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for 
        for (RAJA::Index_type i = domain.fpz ; i <= domain.lpz ; i++ ) {
          VOL3D_CALC_BODY(i);
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

      VOL3D_CALC_DATA;

      NDPTRSET(x,x0,x1,x2,x3,x4,x5,x6,x7) ;
      NDPTRSET(y,y0,y1,y2,y3,y4,y5,y6,y7) ;
      NDPTRSET(z,z0,z1,z2,z3,z4,z5,z6,z7) ;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(0, run_size, [=](int i) {
          VOL3D_CALC_BODY(i);
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

void VOL3D_CALC::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_vol, getRunSize());
}

void VOL3D_CALC::tearDown(VariantID vid)
{
  freeAligned(m_x);
  freeAligned(m_y);
  freeAligned(m_x);
  freeAligned(m_vol);
  
  if (vid == Baseline_CUDA || vid == RAJA_CUDA) {
    // De-allocate device memory here.
  }
}

} // end namespace apps
} // end namespace rajaperf
