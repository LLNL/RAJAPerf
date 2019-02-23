//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
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

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GESUMMV_DATA_SETUP_CPU \
  const Index_type N = m_N; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr A = m_A; \
  ResReal_ptr B = m_B;

  
POLYBENCH_GESUMMV::POLYBENCH_GESUMMV(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GESUMMV, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N = 30;
      run_reps = 10000;
      break;
    case Small:
      m_N = 90;
      run_reps = 1000;
      break;
    case Medium:
      m_N = 250;
      run_reps = 100;
      break;
    case Large:
      m_N = 1300;
      run_reps = 1;
      break;
    case Extralarge:
      m_N = 2800;
      run_reps = 1;
      break;
    default:
      m_N = 1600;
      run_reps = 120;
      break;
  }

  setDefaultSize( m_N * m_N );
  setDefaultReps(run_reps);

  m_alpha = 0.62;
  m_beta = 1.002;
}

POLYBENCH_GESUMMV::~POLYBENCH_GESUMMV() 
{

}

void POLYBENCH_GESUMMV::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_x, m_N, vid);
  allocAndInitDataConst(m_y, m_N, 0.0, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitData(m_B, m_N * m_N, vid);
}

void POLYBENCH_GESUMMV::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      POLYBENCH_GESUMMV_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < N; ++i ) { 
          POLYBENCH_GESUMMV_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_GESUMMV_BODY2;
          }
          POLYBENCH_GESUMMV_BODY3;
        }

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      POLYBENCH_GESUMMV_DATA_SETUP_CPU;

      POLYBENCH_GESUMMV_VIEWS_RAJA;

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

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0), 
                           static_cast<Real_type>(0.0)),

          [=](Index_type /*i*/, Index_type /*j*/, Real_type& tmpdot,
                                                  Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY1_RAJA;
          },
          [=](Index_type i, Index_type j, Real_type& tmpdot,
                                          Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY2_RAJA;
          },
          [=](Index_type i, Index_type /*j*/, Real_type& tmpdot,
                                              Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY3_RAJA;
          }
        );

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      POLYBENCH_GESUMMV_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = 0; i < N; ++i ) {
          POLYBENCH_GESUMMV_BODY1;
          for (Index_type j = 0; j < N; ++j ) {
            POLYBENCH_GESUMMV_BODY2;
          }
          POLYBENCH_GESUMMV_BODY3;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      POLYBENCH_GESUMMV_DATA_SETUP_CPU;

      POLYBENCH_GESUMMV_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple( RAJA::RangeSegment{0, N},
                            RAJA::RangeSegment{0, N} ),
          RAJA::make_tuple(static_cast<Real_type>(0.0),
                           static_cast<Real_type>(0.0)),

          [=](Index_type /*i*/, Index_type /*j*/, Real_type& tmpdot,
                                                  Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY1_RAJA;
          },
          [=](Index_type i, Index_type j, Real_type& tmpdot,
                                          Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY2_RAJA;
          },
          [=](Index_type i, Index_type /*j*/, Real_type& tmpdot,
                                              Real_type& ydot) {
            POLYBENCH_GESUMMV_BODY3_RAJA;
          }
        );

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
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_GESUMMV::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_y, m_N);
}

void POLYBENCH_GESUMMV::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_A);
  deallocData(m_B);
}

} // end namespace polybench
} // end namespace rajaperf
