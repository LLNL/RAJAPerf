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
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_ADI_DATA_SETUP_CPU \
  const Index_type n = m_n; \
  const Index_type tsteps = m_tsteps; \
\
  Real_type DX,DY,DT; \
  Real_type B1,B2; \
  Real_type mul1,mul2; \
  Real_type a,b,c,d,e,f; \
\
  ResReal_ptr U = m_U; \
  ResReal_ptr V = m_V; \
  ResReal_ptr P = m_P; \
  ResReal_ptr Q = m_Q; 

POLYBENCH_ADI::POLYBENCH_ADI(const RunParams& params)
  : KernelBase(rajaperf::Polybench_ADI, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps;
  switch(lsizespec) {
    case Mini:
      m_n=20; m_tsteps=1; 
      run_reps = 10000;
      break;
    case Small:
      m_n=60; m_tsteps=40; 
      run_reps = 500;
      break;
    case Medium:
      m_n=200; m_tsteps=100; 
      run_reps = 20;
      break;
    case Large:
      m_n=1000; m_tsteps=500; 
      run_reps = 1;
      break;
    case Extralarge:
      m_n=2000; m_tsteps=1000; 
      run_reps = 1;
      break;
    default:
      m_n=200; m_tsteps=100; 
      run_reps = 20;
      break;
  }
  setDefaultSize( m_tsteps * 2*m_n*(m_n+m_n) );
  setDefaultReps(run_reps);
}

POLYBENCH_ADI::~POLYBENCH_ADI() 
{

}

void POLYBENCH_ADI::setUp(VariantID vid)
{
  allocAndInitDataConst(m_U, m_n * m_n, 0.0, vid);
  allocAndInitData(m_V, m_n * m_n, vid);
  allocAndInitData(m_P, m_n * m_n, vid);
  allocAndInitData(m_Q, m_n * m_n, vid);
}

void POLYBENCH_ADI::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_ADI_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        POLYBENCH_ADI_BODY1;

        for (Index_type t = 1; t <= tsteps; ++t) { 

          for (Index_type i = 1; i < n-1; ++i) {
            NEW_POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
              NEW_POLYBENCH_ADI_BODY3;
            }  
            NEW_POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
              NEW_POLYBENCH_ADI_BODY5;
            }  
          }

          for (Index_type i = 1; i < n-1; ++i) {
            NEW_POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
              NEW_POLYBENCH_ADI_BODY7;
            }
            NEW_POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
              NEW_POLYBENCH_ADI_BODY9;
            }  
          }

        }  // tstep loop

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_ADI_DATA_SETUP_CPU;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<3>
            >
          >,
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<4>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<5>
            >,
            RAJA::statement::Lambda<6>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<7>
            >
          >    
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        POLYBENCH_ADI_BODY1;

        for (Index_type t = 1; t <= tsteps; ++t) { 

          RAJA::kernel<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY2;
            },
            [=](Index_type i, Index_type j, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY3;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY4;
            },
            [=](Index_type i, Index_type /*j*/, Index_type k) {
              NEW_POLYBENCH_ADI_BODY5;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY6;
            },
            [=](Index_type i, Index_type j, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY7;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY8;
            },
            [=](Index_type i, Index_type /*j*/, Index_type k) {
              NEW_POLYBENCH_ADI_BODY9;
            }
          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_ADI_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        POLYBENCH_ADI_BODY1;

        for (Index_type t = 1; t <= tsteps; ++t) { 

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            NEW_POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
              NEW_POLYBENCH_ADI_BODY3;
            }  
            NEW_POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
              NEW_POLYBENCH_ADI_BODY5;
            }  
          }

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            NEW_POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
              NEW_POLYBENCH_ADI_BODY7;
            }
            NEW_POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
              NEW_POLYBENCH_ADI_BODY9;
            }  
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_ADI_DATA_SETUP_CPU;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<3>
            >
          >,
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<4>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<5>
            >,
            RAJA::statement::Lambda<6>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<7>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        POLYBENCH_ADI_BODY1;

        for (Index_type t = 1; t <= tsteps; ++t) {

          RAJA::kernel<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY2;
            },
            [=](Index_type i, Index_type j, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY3;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY4;
            },
            [=](Index_type i, Index_type /*j*/, Index_type k) {
              NEW_POLYBENCH_ADI_BODY5;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY6;
            },
            [=](Index_type i, Index_type j, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY7;
            },
            [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
              NEW_POLYBENCH_ADI_BODY8;
            },
            [=](Index_type i, Index_type /*j*/, Index_type k) {
              NEW_POLYBENCH_ADI_BODY9;
            }
          );

        }  // tstep loop

      } // run_reps
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
      std::cout << "\nPOLYBENCH_ADI  Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_ADI::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_U, m_n * m_n);
}

void POLYBENCH_ADI::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_U);
  deallocData(m_V);
  deallocData(m_P);
  deallocData(m_Q);
}

} // end namespace polybench
} // end namespace rajaperf
