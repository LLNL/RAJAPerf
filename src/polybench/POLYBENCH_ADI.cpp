//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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
  Real_type DX = 1.0/(Real_type)n; \
  Real_type DY = 1.0/(Real_type)n; \
  Real_type DT = 1.0/(Real_type)tsteps; \
  Real_type B1 = 2.0; \
  Real_type B2 = 1.0; \
  Real_type mul1 = B1 * DT / (DX * DX); \
  Real_type mul2 = B2 * DT / (DY * DY); \
  Real_type a = -mul1 / 2.0; \
  Real_type b = 1.0 + mul1; \
  Real_type c = a; \
  Real_type d = -mul2 /2.0; \
  Real_type e = 1.0 + mul2; \
  Real_type f = d; \
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

  POLYBENCH_ADI_DATA_SETUP_CPU;

  auto poly_adi_base_lam2 = [=](Index_type i) {
                              POLYBENCH_ADI_BODY2;
                            };
  auto poly_adi_base_lam3 = [=](Index_type i, Index_type j) {
                              POLYBENCH_ADI_BODY3;
                            };
  auto poly_adi_base_lam4 = [=](Index_type i) {
                              POLYBENCH_ADI_BODY4;
                            };
  auto poly_adi_base_lam5 = [=](Index_type i, Index_type k) {
                              POLYBENCH_ADI_BODY5;
                            };
  auto poly_adi_base_lam6 = [=](Index_type i) {
                              POLYBENCH_ADI_BODY6;
                            };
  auto poly_adi_base_lam7 = [=](Index_type i, Index_type j) {
                              POLYBENCH_ADI_BODY7;
                            };
  auto poly_adi_base_lam8 = [=](Index_type i) {
                              POLYBENCH_ADI_BODY8;
                            };
  auto poly_adi_base_lam9 = [=](Index_type i, Index_type k) {
                              POLYBENCH_ADI_BODY9;
                            };

  POLYBENCH_ADI_VIEWS_RAJA;

  auto poly_adi_lam2 = [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY2_RAJA;
                       };
  auto poly_adi_lam3 = [=](Index_type i, Index_type j, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY3_RAJA;
                       };
  auto poly_adi_lam4 = [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY4_RAJA;
                       };
  auto poly_adi_lam5 = [=](Index_type i, Index_type /*j*/, Index_type k) {
                         POLYBENCH_ADI_BODY5_RAJA;
                       };
  auto poly_adi_lam6 = [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY6_RAJA;
                       };
  auto poly_adi_lam7 = [=](Index_type i, Index_type j, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY7_RAJA;
                       };
  auto poly_adi_lam8 = [=](Index_type i, Index_type /*j*/, Index_type /*k*/) {
                         POLYBENCH_ADI_BODY8_RAJA;
                       };
  auto poly_adi_lam9 = [=](Index_type i, Index_type /*j*/, Index_type k) {
                         POLYBENCH_ADI_BODY9_RAJA;
                       };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) { 

          for (Index_type i = 1; i < n-1; ++i) {
            POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY3;
            }  
            POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY5;
            }  
          }

          for (Index_type i = 1; i < n-1; ++i) {
            POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY7;
            }
            POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY9;
            }  
          }

        }  // tstep loop

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
            RAJA::statement::Lambda<2>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<3>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) { 

          RAJA::kernel<EXEC_POL>( 
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam2,
            poly_adi_lam3,
            poly_adi_lam4,
            poly_adi_lam5

          );

          RAJA::kernel<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam6,
            poly_adi_lam7,
            poly_adi_lam8,
            poly_adi_lam9

          );

        }  // tstep loop

      } // run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) { 

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            POLYBENCH_ADI_BODY2;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY3;
            }  
            POLYBENCH_ADI_BODY4;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY5;
            }  
          }

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            POLYBENCH_ADI_BODY6;
            for (Index_type j = 1; j < n-1; ++j) {
              POLYBENCH_ADI_BODY7;
            }
            POLYBENCH_ADI_BODY8;
            for (Index_type k = n-2; k >= 1; --k) {
              POLYBENCH_ADI_BODY9;
            }  
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case OpenMP_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) {

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            poly_adi_base_lam2(i);
            for (Index_type j = 1; j < n-1; ++j) {
              poly_adi_base_lam3(i, j);
            }
            poly_adi_base_lam4(i);
            for (Index_type k = n-2; k >= 1; --k) {
              poly_adi_base_lam5(i, k);
            }
          }

          #pragma omp parallel for
          for (Index_type i = 1; i < n-1; ++i) {
            poly_adi_base_lam6(i);
            for (Index_type j = 1; j < n-1; ++j) {
              poly_adi_base_lam7(i, j);
            }
            poly_adi_base_lam8(i);
            for (Index_type k = n-2; k >= 1; --k) {
              poly_adi_base_lam9(i, k);
            }
          }

        }  // tstep loop

      }  // run_reps
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

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
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 1; t <= tsteps; ++t) {

          RAJA::kernel<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam2,
            poly_adi_lam3,
            poly_adi_lam4,
            poly_adi_lam5

          );

          RAJA::kernel<EXEC_POL>(
            RAJA::make_tuple(RAJA::RangeSegment{1, n-1},
                             RAJA::RangeSegment{1, n-1},
                             RAJA::RangeStrideSegment{n-2, 0, -1}),

            poly_adi_lam6,
            poly_adi_lam7,
            poly_adi_lam8,
            poly_adi_lam9

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
