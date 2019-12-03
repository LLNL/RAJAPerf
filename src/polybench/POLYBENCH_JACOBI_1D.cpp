//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_1D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>


namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_JACOBI_1D_DATA_SETUP_CPU \
  ResReal_ptr A = m_Ainit; \
  ResReal_ptr B = m_Binit; \
  const Index_type N = m_N; \
  const Index_type tsteps = m_tsteps;

#define POLYBENCH_JACOBI_1D_DATA_RESET_CPU \
  m_Ainit = m_A; \
  m_Binit = m_B; \
  m_A = A; \
  m_B = B; 
  
POLYBENCH_JACOBI_1D::POLYBENCH_JACOBI_1D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_1D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=300;
      m_tsteps=20;
      run_reps = 10000;
      break;
    case Small:
      m_N=1200;
      m_tsteps=100;
      run_reps = 1000;
      break;
    case Medium:
      m_N=4000;
      m_tsteps=100;
      run_reps = 100;
      break;
    case Large:
      m_N=200000;
      m_tsteps=50;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=2000000;
      m_tsteps=10;
      run_reps = 20;
      break;
    default:
      m_N=4000000;
      m_tsteps=10;
      run_reps = 10;
      break;
  }

  setDefaultSize( m_tsteps * 2 * m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_JACOBI_1D::~POLYBENCH_JACOBI_1D() 
{

}

void POLYBENCH_JACOBI_1D::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N, vid);
  allocAndInitData(m_Binit, m_N, vid);
  allocAndInitDataConst(m_A, m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N, 0.0, vid);
}

void POLYBENCH_JACOBI_1D::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_JACOBI_1D_DATA_SETUP_CPU;

  auto poly_jacobi1d_lam1 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY1;
                            };
  auto poly_jacobi1d_lam2 = [=] (Index_type i) {
                              POLYBENCH_JACOBI_1D_BODY2;
                            };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) { 
          for (Index_type i = 1; i < N-1; ++i ) { 
            POLYBENCH_JACOBI_1D_BODY1;
          }
          for (Index_type i = 1; i < N-1; ++i ) { 
            POLYBENCH_JACOBI_1D_BODY2;
          }
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }


#if defined(RUN_RAJA_SEQ)      
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::loop_exec> ( RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam1
          );

          RAJA::forall<RAJA::loop_exec> ( RAJA::RangeSegment{1, N-1}, 
            poly_jacobi1d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY1;
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            POLYBENCH_JACOBI_1D_BODY2;
          }
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            poly_jacobi1d_lam1(i);
          }
          #pragma omp parallel for
          for (Index_type i = 1; i < N-1; ++i ) {
            poly_jacobi1d_lam2(i);
          }
        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam1
          );

          RAJA::forall<RAJA::omp_parallel_for_exec> (RAJA::RangeSegment{1, N-1},
            poly_jacobi1d_lam2
          );

        }

      }
      stopTimer();

      POLYBENCH_JACOBI_1D_DATA_RESET_CPU;

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
      std::cout << "\n  POLYBENCH_JACOBI_1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_JACOBI_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_A, m_N);
  checksum[vid] += calcChecksum(m_B, m_N);
}

void POLYBENCH_JACOBI_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
