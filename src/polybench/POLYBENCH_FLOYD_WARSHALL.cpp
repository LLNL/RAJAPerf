//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE

//#define USE_RAJA_OMP_COLLAPSE
#undef USE_RAJA_OMP_COLLAPSE

namespace rajaperf 
{
namespace polybench
{

 
POLYBENCH_FLOYD_WARSHALL::POLYBENCH_FLOYD_WARSHALL(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FLOYD_WARSHALL, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=60;
      run_reps = 100000;
      break;
    case Small:
      m_N=180;
      run_reps = 1000;
      break;
    case Medium:
      m_N=500;
      run_reps = 100;
      break;
    case Large:
      m_N=2800;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=5600;
      run_reps = 1;
      break;
    default:
      m_N=300;
      run_reps = 60;
      break;
  }

  setDefaultSize( m_N*m_N*m_N );
  setDefaultReps(run_reps);
}

POLYBENCH_FLOYD_WARSHALL::~POLYBENCH_FLOYD_WARSHALL() 
{

}

void POLYBENCH_FLOYD_WARSHALL::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitDataRandSign(m_pin, m_N*m_N, vid);
  allocAndInitDataConst(m_pout, m_N*m_N, 0.0, vid);
}

void POLYBENCH_FLOYD_WARSHALL::runKernel(VariantID vid)
{
  const Index_type run_reps= getRunReps();

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  auto poly_floydwarshall_base_lam = [=](Index_type k, Index_type i, 
                                         Index_type j) {
                                       POLYBENCH_FLOYD_WARSHALL_BODY;
                                     };

  POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA; 

  auto poly_floydwarshall_lam = [=](Index_type k, Index_type i, Index_type j) {
                                  POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
                                };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) { 
          for (Index_type i = 0; i < N; ++i) { 
            for (Index_type j = 0; j < N; ++j) { 
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }
          }
        }

      }
      stopTimer();

      break;
    }


#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
          for (Index_type i = 0; i < N; ++i) {
            for (Index_type j = 0; j < N; ++j) {
              poly_floydwarshall_base_lam(k, i, j);
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          poly_floydwarshall_lam 
        );

      }
      stopTimer();

      break;
    }

#endif // RUN_RAJA_SEQ


#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(2)
#else
          #pragma omp parallel for
#endif
          for (Index_type i = 0; i < N; ++i) {  
            for (Index_type j = 0; j < N; ++j) {
              POLYBENCH_FLOYD_WARSHALL_BODY;
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {
#if defined(USE_OMP_COLLAPSE)
          #pragma omp parallel for collapse(2)
#else
          #pragma omp parallel for
#endif
          for (Index_type i = 0; i < N; ++i) {
            for (Index_type j = 0; j < N; ++j) {
              poly_floydwarshall_base_lam(k, i, j);
            }
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

#if defined(USE_RAJA_OMP_COLLAPSE)
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                      RAJA::ArgList<1, 2>,
              RAJA::statement::Lambda<0>
            >
          >
        >;
#else
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N},
                                                 RAJA::RangeSegment{0, N}),
          poly_floydwarshall_lam 
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
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown variant id = " << vid << std::endl;
    }

  }

}

void POLYBENCH_FLOYD_WARSHALL::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_pout, m_N*m_N);
}

void POLYBENCH_FLOYD_WARSHALL::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_pin);
  deallocData(m_pout);
}

} // end namespace polybench
} // end namespace rajaperf
