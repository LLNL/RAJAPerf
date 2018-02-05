//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define DEL_DOT_VEC_2D_DATA_SETUP_CPU \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr xdot = m_xdot; \
  ResReal_ptr ydot = m_ydot; \
  ResReal_ptr div = m_div; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  ResReal_ptr x1,x2,x3,x4 ; \
  ResReal_ptr y1,y2,y3,y4 ; \
  ResReal_ptr fx1,fx2,fx3,fx4 ; \
  ResReal_ptr fy1,fy2,fy3,fy4 ;


DEL_DOT_VEC_2D::DEL_DOT_VEC_2D(const RunParams& params)
  : KernelBase(rajaperf::Apps_DEL_DOT_VEC_2D, params)
{
  setDefaultSize(312);  // See rzmax in ADomain struct
  setDefaultReps(1050);

  m_domain = new ADomain(getRunSize(), /* ndims = */ 2);

  m_array_length = m_domain->nnalls;
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
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_y, m_array_length, 0.0, vid);

  Real_type dx = 0.2;
  Real_type dy = 0.1;
  setMeshPositions_2d(m_x, dx, m_y, dy, *m_domain);

  allocAndInitData(m_xdot, m_array_length, vid);
  allocAndInitData(m_ydot, m_array_length, vid);

  allocAndInitDataConst(m_div, m_array_length, 0.0, vid);

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

      DEL_DOT_VEC_2D_DATA_SETUP_CPU;
      DEL_DOT_VEC_2D_DATA_INDEX;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          DEL_DOT_VEC_2D_BODY_INDEX;
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      DEL_DOT_VEC_2D_DATA_SETUP_CPU;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      RAJA::ListSegment zones(m_domain->real_zones, m_domain->n_real_zones);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(zones, [=](Index_type i) {
          DEL_DOT_VEC_2D_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      DEL_DOT_VEC_2D_DATA_SETUP_CPU;
      DEL_DOT_VEC_2D_DATA_INDEX;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for 
        for (Index_type ii = ibegin ; ii < iend ; ++ii ) {
          DEL_DOT_VEC_2D_BODY_INDEX;
          DEL_DOT_VEC_2D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      DEL_DOT_VEC_2D_DATA_SETUP_CPU;

      NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
      NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
      NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
      NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

      RAJA::ListSegment zones(m_domain->real_zones, m_domain->n_real_zones);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(zones, [=](Index_type i) { 
          DEL_DOT_VEC_2D_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  DEL_DOT_VEC_2D : Unknown variant id = " << vid << std::endl;
    }

  }
}

void DEL_DOT_VEC_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_div, m_array_length);
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
