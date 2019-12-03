//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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
  ResReal_ptr A = m_A; \
  const Index_type N = m_N;

  
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

  POLYBENCH_MVT_DATA_SETUP_CPU;

  auto poly_mvt_base_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                              POLYBENCH_MVT_BODY2;
                             };
  auto poly_mvt_base_lam3 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_MVT_BODY3;
                            };
  auto poly_mvt_base_lam5 = [=] (Index_type i, Index_type j, Real_type &dot) {
                              POLYBENCH_MVT_BODY5;
                            };
  auto poly_mvt_base_lam6 = [=] (Index_type i, Real_type &dot) {
                              POLYBENCH_MVT_BODY6;
                            };

  POLYBENCH_MVT_VIEWS_RAJA;

  auto poly_mvt_lam1 = [=] (Index_type /* i */, Index_type /* j */, 
                            Real_type &dot) {
                            POLYBENCH_MVT_BODY1_RAJA;
                           };
  auto poly_mvt_lam2 = [=] (Index_type i, Index_type j, Real_type &dot) {
                            POLYBENCH_MVT_BODY2_RAJA;
                           };
  auto poly_mvt_lam3 = [=] (Index_type i, Index_type /* j */, Real_type &dot) {
                            POLYBENCH_MVT_BODY3_RAJA;
                           };
  auto poly_mvt_lam4 = [=] (Index_type /* i */, Index_type /* j */, 
                            Real_type &dot) {
                            POLYBENCH_MVT_BODY4_RAJA;
                           };
  auto poly_mvt_lam5 = [=] (Index_type i, Index_type j, Real_type &dot) {
                            POLYBENCH_MVT_BODY5_RAJA;
                           };
  auto poly_mvt_lam6 = [=] (Index_type i, Index_type /* j */, Real_type &dot) {
                            POLYBENCH_MVT_BODY6_RAJA;
                           };

  switch ( vid ) {

    case Base_Seq : {

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
 
            poly_mvt_lam1,
            poly_mvt_lam2,
            poly_mvt_lam3
 
          );

          RAJA::kernel_param<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),
 
            poly_mvt_lam4,
            poly_mvt_lam5, 
            poly_mvt_lam6
 
          );

        }); // end sequential region (for single-source code)

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

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

    case OpenMP_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for nowait
          for (Index_type i = 0; i < N; ++i ) {
            POLYBENCH_MVT_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              poly_mvt_base_lam2(i, j, dot);
            }
            poly_mvt_base_lam3(i, dot);
          }

          #pragma omp for nowait
          for (Index_type i = 0; i < N; ++i ) {
            POLYBENCH_MVT_BODY4;
            for (Index_type j = 0; j < N; ++j ) {
              poly_mvt_base_lam5(i, j, dot);
            }
            poly_mvt_base_lam6(i, dot);
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

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
 
            poly_mvt_lam1,
            poly_mvt_lam2,
            poly_mvt_lam3
 
          );

          RAJA::kernel_param<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{0, N},
                             RAJA::RangeSegment{0, N}),
            RAJA::make_tuple(static_cast<Real_type>(0.0)),
 
            poly_mvt_lam4,
            poly_mvt_lam5, 
            poly_mvt_lam6
 
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
