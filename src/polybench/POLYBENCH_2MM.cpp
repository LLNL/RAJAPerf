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

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"
#include "common/DataUtils.hpp"


#include <iostream>
#include <cstring>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_2MM_DATA_SETUP_CPU \
  ResReal_ptr tmp = m_tmp; \
  ResReal_ptr A = m_A; \
  ResReal_ptr B = m_B; \
  ResReal_ptr C = m_C; \
  ResReal_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; 

  
POLYBENCH_2MM::POLYBENCH_2MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_2MM, params)
{
  m_alpha = 1.5;
  m_beta = 1.2;
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=22; m_nl=24;
      m_run_reps = 10000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=70; m_nl=80;
      m_run_reps = 1000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      m_run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1100; m_nl=1200;
      m_run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2200; m_nl=2400;
      m_run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=210; m_nl=220;
      m_run_reps = 100;
      break;
  }

  setDefaultSize( m_ni*m_nj*(1+m_nk) + m_ni*m_nl*(1+m_nj) );
  setDefaultReps(m_run_reps);
}

POLYBENCH_2MM::~POLYBENCH_2MM() 
{

}

void POLYBENCH_2MM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_tmp, m_ni * m_nj, vid);
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nl, vid);
  allocAndInitDataConst(m_D, m_ni * m_nl, 0.0, vid);
  allocAndInitData(m_DD, m_ni * m_nl, vid);
}

void POLYBENCH_2MM::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_2MM_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; i++ ) { 
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_2MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_2MM_BODY2;
            }
          }
        }

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));

        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_2MM_BODY3;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_2MM_BODY4;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_2MM_DATA_SETUP_CPU;

      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::loop_exec>,
        RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nj)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_2MM_BODY1;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nk}, [=] (int k) {
              POLYBENCH_2MM_BODY2; 
            });
        });

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nl)),
            [=](Index_type i, Index_type l) {     
            POLYBENCH_2MM_BODY3;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nj}, [=] (int j) {
              POLYBENCH_2MM_BODY4; 
            });
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      POLYBENCH_2MM_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for collapse(2) 
        for (Index_type i = 0; i < ni; i++ ) {
          for(Index_type j = 0; j < nj; j++) {
            POLYBENCH_2MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_2MM_BODY2;
            }
          }
        }

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));

        #pragma omp parallel for collapse(2)  
        for(Index_type i = 0; i < ni; i++) {
          for(Index_type l = 0; l < nl; l++) {
            POLYBENCH_2MM_BODY3;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_2MM_BODY4;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_2MM_DATA_SETUP_CPU;

      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::omp_parallel_for_exec>,
        RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nj)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_2MM_BODY1;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nk}, [=] (int k) {
              POLYBENCH_2MM_BODY2; 
            });
        });

        memcpy(m_D,m_DD,m_ni * m_nl * sizeof(Real_type));

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nl)),
            [=](Index_type i, Index_type l) {     
            POLYBENCH_2MM_BODY3;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nj}, [=] (int j) {
              POLYBENCH_2MM_BODY4; 
            });
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
      std::cout << "\n  POLYBENCH_2MM : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_2MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_D, m_ni * m_nl);
}

void POLYBENCH_2MM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_tmp);
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
  deallocData(m_DD);
}

} // end namespace polybench
} // end namespace rajaperf
