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

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_MVT_DATA_SETUP_CPU \
  ResReal_ptr x1 = m_x1; \
  ResReal_ptr x2 = m_x2; \
  ResReal_ptr y1 = m_y1; \
  ResReal_ptr y2 = m_y2; \
  ResReal_ptr A = m_A;

  
POLYBENCH_MVT::POLYBENCH_MVT(const RunParams& params)
  : KernelBase(rajaperf::Polybench_MVT, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=40;
      run_reps = 10000;
      break;
    case Small:
      m_N=120;
      run_reps = 1000;
      break;
    case Medium:
      m_N=1000;
      run_reps = 100;
      break;
    case Large:
      m_N=2000;
      run_reps = 40;
      break;
    case Extralarge:
      m_N=4000;
      run_reps = 10;
      break;
    default:
      m_N=4000;
      run_reps = 10;
      break;
  }

  setDefaultSize( 2*m_N*m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_MVT::~POLYBENCH_MVT() 
{

}

void POLYBENCH_MVT::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_y1, m_N, vid);
  allocAndInitData(m_y2, m_N, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitDataConst(m_x1, m_N, 0.0, vid);
  allocAndInitDataConst(m_x2, m_N, 0.0, vid);
}

void POLYBENCH_MVT::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();
  const Index_type N = m_N;

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_MVT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < N; ++i ) { 
          POLYBENCH_MVT_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_MVT_BODY2;
          }
          POLYBENCH_MVT_BODY3;
        }

        for (Index_type i = 0; i < N; ++i ) { 
          POLYBENCH_MVT_BODY4;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_MVT_BODY5;
          }
          POLYBENCH_MVT_BODY6;
        }

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_MVT_DATA_SETUP_CPU;

      POLYBENCH_MVT_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::seq_region>( [=]() {

          RAJA::kernel_param<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),
  
            [=] (Index_type /* i */, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY1_RAJA;
            },
            [=] (Index_type i, Index_type j, Real_type &dot) {
              POLYBENCH_MVT_BODY2_RAJA;
            },
            [=] (Index_type i, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY3_RAJA;
            }
 
          );

          RAJA::kernel_param<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),
  
            [=] (Index_type /* i */, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY4_RAJA;
            },
            [=] (Index_type i, Index_type j, Real_type &dot) {
              POLYBENCH_MVT_BODY5_RAJA;
            },
            [=] (Index_type i, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY6_RAJA;
            }
 
          );

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_MVT_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for nowait
          for (Index_type i = 0; i < N; ++i ) { 
            POLYBENCH_MVT_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY2;
            }
            POLYBENCH_MVT_BODY3;
          }

          #pragma omp for nowait
          for (Index_type i = 0; i < N; ++i ) { 
            POLYBENCH_MVT_BODY4;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY5;
            }
            POLYBENCH_MVT_BODY6;
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_MVT_DATA_SETUP_CPU;

      POLYBENCH_MVT_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_for_nowait_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::kernel_param<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),

            [=] (Index_type /* i */, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY1_RAJA;
            },
            [=] (Index_type i, Index_type j, Real_type &dot) {
              POLYBENCH_MVT_BODY2_RAJA;
            },
            [=] (Index_type i, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY3_RAJA;
            }

          );

          RAJA::kernel_param<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),

            [=] (Index_type /* i */, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY4_RAJA;
            },
            [=] (Index_type i, Index_type j, Real_type &dot) {
              POLYBENCH_MVT_BODY5_RAJA;
            },
            [=] (Index_type i, Index_type /* j */, Real_type &dot) {
              POLYBENCH_MVT_BODY6_RAJA;
            }

          );

        }); // end omp parallel region

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
      std::cout << "\n  POLYBENCH_MVT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_MVT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x1, m_N);
  checksum[vid] += calcChecksum(m_x2, m_N);
}

void POLYBENCH_MVT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x1);
  deallocData(m_x2);
  deallocData(m_y1);
  deallocData(m_y2);
  deallocData(m_A);
}

} // end namespace polybench
} // end namespace rajaperf
