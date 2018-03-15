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

#include "POLYBENCH_GEMMVER.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMMVER_DATA_SETUP_CPU \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  ResReal_ptr A = m_A; \
  ResReal_ptr u1 = m_u1; \
  ResReal_ptr v1 = m_v1; \
  ResReal_ptr u2 = m_u2; \
  ResReal_ptr v2 = m_v2; \
  ResReal_ptr w = m_w; \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr z = m_z; 

  
POLYBENCH_GEMMVER::POLYBENCH_GEMMVER(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMMVER, params)
{
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_n=40;
      run_reps = 200;
      break;
    case Small:
      m_n=120; 
      run_reps = 200;
      break;
    case Medium:
      m_n=400;
      run_reps = 20;
      break;
    case Large:
      m_n=2000;
      run_reps = 20;
      break;
    case Extralarge:
      m_n=4000; 
      run_reps = 5;
      break;
    default:
      m_n=400;
      run_reps = 20;
      break;
  }

  setDefaultSize(m_n*m_n + m_n*m_n + m_n + m_n*m_n);
  setDefaultReps(run_reps);

  m_alpha = 1.5;
  m_beta = 1.2;
}

POLYBENCH_GEMMVER::~POLYBENCH_GEMMVER() 
{
}

void POLYBENCH_GEMMVER::setUp(VariantID vid)
{
  (void) vid;

  allocAndInitData(m_A, m_n * m_n, vid);
  allocAndInitData(m_u1, m_n, vid);
  allocAndInitData(m_v1, m_n, vid);
  allocAndInitData(m_u2, m_n, vid);
  allocAndInitData(m_v2, m_n, vid);
  allocAndInitDataConst(m_w, m_n, 0.0, vid);
  allocAndInitData(m_x, m_n, vid);
  allocAndInitData(m_y, m_n, vid);
  allocAndInitData(m_z, m_n, vid);

}

void POLYBENCH_GEMMVER::runKernel(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type n = m_n;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_GEMMVER_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }
        }

        for (Index_type i = 0; i < n; i++ ) { 
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }
        }

        for (Index_type i = 0; i < n; i++ ) { 
          POLYBENCH_GEMMVER_BODY3;
        }

        for (Index_type i = 0; i < n; i++ ) { 
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_GEMMVER_DATA_SETUP_CPU;

      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::loop_exec>,
        RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY2;
        });

        RAJA::forall<RAJA::loop_exec> (
          RAJA::RangeSegment{0, n}, [=] (int i) {
          POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY4;
        });


      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      POLYBENCH_GEMMVER_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY1;
          }
        }


        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY2;
          }
        } 

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          POLYBENCH_GEMMVER_BODY3;
        }

        #pragma omp parallel for  
        for (Index_type i = 0; i < n; i++ ) {
          for (Index_type j = 0; j < n; j++) {
            POLYBENCH_GEMMVER_BODY4;
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_GEMMVER_DATA_SETUP_CPU;

      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::loop_exec>,
        RAJA::nested::For<0, RAJA::omp_parallel_for_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY1;
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY2;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec> (
          RAJA::RangeSegment{0, n}, [=] (int i) {
          POLYBENCH_GEMMVER_BODY3; 
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, n),
                           RAJA::RangeSegment(0, n)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_GEMMVER_BODY4;
        });

      }
      stopTimer();

      break;
    }
#endif //RAJA_ENABLE_OPENMP

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
      std::cout << "\n  POLYBENCH_GEMMVER : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_GEMMVER::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, m_n);
}

void POLYBENCH_GEMMVER::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_u1);
  deallocData(m_v1);
  deallocData(m_u2);
  deallocData(m_v2);
  deallocData(m_w);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace basic
} // end namespace rajaperf
