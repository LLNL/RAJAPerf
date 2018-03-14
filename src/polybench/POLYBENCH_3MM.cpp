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

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_3MM_DATA_SETUP_CPU \
  ResReal_ptr A = m_A; \
  ResReal_ptr B = m_B; \
  ResReal_ptr C = m_C; \
  ResReal_ptr D = m_D; \
  ResReal_ptr E = m_E; \
  ResReal_ptr F = m_F; \
  ResReal_ptr G = m_G; 
  
  
POLYBENCH_3MM::POLYBENCH_3MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_3MM, params)
{
  SizeSpec_T lsizespec = KernelBase::getSizeSpec();
  switch(lsizespec) {
    case Mini:
      m_ni=16; m_nj=18; m_nk=20; m_nl=22; m_nm=24;
      m_run_reps = 100000;
      break;
    case Small:
      m_ni=40; m_nj=50; m_nk=60; m_nl=70; m_nm=80;
      m_run_reps = 5000;
      break;
    case Medium:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
    case Large:
      m_ni=800; m_nj=900; m_nk=1000; m_nl=1100; m_nm=1200;
      m_run_reps = 1;
      break;
    case Extralarge:
      m_ni=1600; m_nj=1800; m_nk=2000; m_nl=2200; m_nm=2400;
      m_run_reps = 1;
      break;
    default:
      m_ni=180; m_nj=190; m_nk=200; m_nl=210; m_nm=220;
      m_run_reps = 100;
      break;
  }

  setDefaultSize(m_ni*m_nj*(1+m_nk) + m_nj*m_nl*(1+m_nm) + m_ni*m_nl*(1+m_nj));
  setDefaultReps(m_run_reps);


}

POLYBENCH_3MM::~POLYBENCH_3MM() 
{
}

void POLYBENCH_3MM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nm, vid);
  allocAndInitData(m_D, m_nm * m_nl, vid);
  allocAndInitData(m_E, m_ni * m_nj, vid);
  allocAndInitData(m_F, m_nj * m_nl, vid);
  allocAndInitDataConst(m_G, m_ni * m_nl, 0.0, vid);
}

void POLYBENCH_3MM::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ni = m_ni;
  const Index_type nj = m_nj;
  const Index_type nk = m_nk;
  const Index_type nl = m_nl;
  const Index_type nm = m_nm;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_3MM_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; i++ ) {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
          }
        }

        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY3;
            for (Index_type m = 0; m < nm; m++) {
              POLYBENCH_3MM_BODY4;
            }
          }
        }

        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY5;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_3MM_BODY6;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      POLYBENCH_3MM_DATA_SETUP_CPU;
      
      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::loop_exec>,
        RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nj)),
            [=](Index_type i, Index_type j) {     
            POLYBENCH_3MM_BODY1;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nk}, [=] (int k) {
              POLYBENCH_3MM_BODY2; 
            });
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, nj),
                           RAJA::RangeSegment(0, nl)),
            [=](Index_type j, Index_type l) {     
            POLYBENCH_3MM_BODY3;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nm}, [=] (int m) {
              POLYBENCH_3MM_BODY4; 
            });
        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                           RAJA::RangeSegment(0, nl)),
            [=](Index_type i, Index_type l) {     
            POLYBENCH_3MM_BODY5;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nj}, [=] (int j) {
              POLYBENCH_3MM_BODY6; 
            });
        });

      } // end run_reps
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      POLYBENCH_3MM_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for  
        for (Index_type i = 0; i < ni; i++ )  {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
          }
        }

        #pragma omp parallel for   
        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY3;
            for (Index_type m = 0; m < nm; m++) {
              POLYBENCH_3MM_BODY4;
            }
          }
        }

        #pragma omp parallel for   
        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY5;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_3MM_BODY6;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_3MM_DATA_SETUP_CPU;

      using EXEC_POL = RAJA::nested::Policy<
        RAJA::nested::For<1, RAJA::omp_parallel_for_exec>,
        RAJA::nested::For<0, RAJA::loop_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple( RAJA::RangeSegment{0, ni}, 
                            RAJA::RangeSegment{0, nj}),
          [=] (int i, int j) {

          POLYBENCH_3MM_BODY1;

          RAJA::forall<RAJA::loop_exec> (
          RAJA::RangeSegment{0, nk}, [=] (int k) {
            POLYBENCH_3MM_BODY2; 
          });

        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple( RAJA::RangeSegment{0, nj}, 
                            RAJA::RangeSegment{0, nl}),
          [=] (int j, int l) {

          POLYBENCH_3MM_BODY3;

          RAJA::forall<RAJA::loop_exec> (
          RAJA::RangeSegment{0, nm}, [=] (int m) {
            POLYBENCH_3MM_BODY4;
          });

        });

        RAJA::nested::forall(EXEC_POL{},
          RAJA::make_tuple( RAJA::RangeSegment{0, ni}, 
                            RAJA::RangeSegment{0, nl}),
          [=] (int i, int l) {

            POLYBENCH_3MM_BODY5;

            RAJA::forall<RAJA::loop_exec> (
              RAJA::RangeSegment{0, nj}, [=] (int j) {
              POLYBENCH_3MM_BODY6;
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

void POLYBENCH_3MM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_G, m_ni * m_nl);
}

void POLYBENCH_3MM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
  deallocData(m_E);
  deallocData(m_F);
  deallocData(m_G);
}

} // end namespace basic
} // end namespace rajaperf
