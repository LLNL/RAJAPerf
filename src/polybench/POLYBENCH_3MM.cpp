//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>


#define USE_OMP_COLLAPSE
//#undef USE_OMP_COLLAPSE

#define USE_RAJA_OMP_COLLAPSE
//#undef USE_RAJA_OMP_COLLAPSE


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
  SizeSpec lsizespec = KernelBase::getSizeSpec();
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
  allocAndInitDataConst(m_E, m_ni * m_nj, 0.0, vid);
  allocAndInitDataConst(m_F, m_nj * m_nl, 0.0, vid);
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

  POLYBENCH_3MM_DATA_SETUP_CPU;

  auto poly_3mm_base_lam2 = [=] (Index_type i, Index_type j, Index_type k,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY2;
                            };
  auto poly_3mm_base_lam3 = [=] (Index_type i, Index_type j,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY3;
                            };
  auto poly_3mm_base_lam5 = [=] (Index_type j, Index_type l, Index_type m,
                                 Real_type &dot) {
                               POLYBENCH_3MM_BODY5;
                            };
  auto poly_3mm_base_lam6 = [=] (Index_type j, Index_type l,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY6;
                            };
  auto poly_3mm_base_lam8 = [=] (Index_type i, Index_type l, Index_type j,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY8;
                            };
  auto poly_3mm_base_lam9 = [=] (Index_type i, Index_type l,
                                 Real_type &dot) {
                              POLYBENCH_3MM_BODY9;
                            };

  POLYBENCH_3MM_VIEWS_RAJA;

  auto poly_3mm_lam1 = [=] (Index_type /*i*/, Index_type /*j*/, Index_type /*k*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY1_RAJA;
                            };
  auto poly_3mm_lam2 = [=] (Index_type i, Index_type j, Index_type k, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY2_RAJA;
                            };
  auto poly_3mm_lam3 = [=] (Index_type i, Index_type j, Index_type /*k*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY3_RAJA;
                            };

  auto poly_3mm_lam4 = [=] (Index_type /*j*/, Index_type /*l*/, Index_type /*m*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY4_RAJA;
                            };
  auto poly_3mm_lam5 = [=] (Index_type j, Index_type l, Index_type m, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY5_RAJA;
                            };
  auto poly_3mm_lam6 = [=] (Index_type j, Index_type l, Index_type /*m*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY6_RAJA;
                            };
  auto poly_3mm_lam7 = [=] (Index_type /*i*/, Index_type /*l*/, Index_type /*j*/,
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY7_RAJA;
                            };
  auto poly_3mm_lam8 = [=] (Index_type i, Index_type l, Index_type j, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY8_RAJA;
                            };
  auto poly_3mm_lam9 = [=] (Index_type i, Index_type l, Index_type /*j*/, 
                            Real_type &dot) {
                              POLYBENCH_3MM_BODY9_RAJA;
                            };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = 0; i < ni; i++ ) {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
            POLYBENCH_3MM_BODY3;
          }
        }

        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY4;
            for (Index_type m = 0; m < nm; m++) {
              POLYBENCH_3MM_BODY5;
            }
            POLYBENCH_3MM_BODY6;
          }
        }

        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY7;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_3MM_BODY8;
            }
            POLYBENCH_3MM_BODY9;
          }
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
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >,
              RAJA::statement::Lambda<2>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam1,
          poly_3mm_lam2,
          poly_3mm_lam3

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam4,
          poly_3mm_lam5,
          poly_3mm_lam6

        ); 

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam7,
          poly_3mm_lam8,
          poly_3mm_lam9

        );

      } // end run_reps
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)                        
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type i = 0; i < ni; i++ )  {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              POLYBENCH_3MM_BODY2;
            }
            POLYBENCH_3MM_BODY3;
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY4;
            for (Index_type m = 0; m < nm; m++) {
              POLYBENCH_3MM_BODY5;
            }
            POLYBENCH_3MM_BODY6;
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for  
#endif
        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY7;
            for (Index_type j = 0; j < nj; j++) {
              POLYBENCH_3MM_BODY8;
            }
            POLYBENCH_3MM_BODY9;
          }
        }

      }
      stopTimer();

      break;
    }

    case OpenMP_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type i = 0; i < ni; i++ )  {
          for (Index_type j = 0; j < nj; j++) {
            POLYBENCH_3MM_BODY1;
            for (Index_type k = 0; k < nk; k++) {
              poly_3mm_base_lam2(i, j, k, dot);
            }
            poly_3mm_base_lam3(i, j, dot);
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type j = 0; j < nj; j++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY4;
            for (Index_type m = 0; m < nm; m++) {
              poly_3mm_base_lam5(j, l, m, dot);
            }
            poly_3mm_base_lam6(j, l, dot);
          }
        }

#if defined(USE_OMP_COLLAPSE)
        #pragma omp parallel for collapse(2)
#else
        #pragma omp parallel for
#endif
        for (Index_type i = 0; i < ni; i++) {
          for (Index_type l = 0; l < nl; l++) {
            POLYBENCH_3MM_BODY7;
            for (Index_type j = 0; j < nj; j++) {
              poly_3mm_base_lam8(i, l, j, dot);
            }
            poly_3mm_base_lam9(i, l, dot);
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
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<0, 1>,
            RAJA::statement::Lambda<0>,
            RAJA::statement::For<2, RAJA::loop_exec,
              RAJA::statement::Lambda<1>
            >,
            RAJA::statement::Lambda<2>
          >
        >;
#else
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
            RAJA::statement::For<1, RAJA::loop_exec,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::loop_exec,
                RAJA::statement::Lambda<1>
              >,
              RAJA::statement::Lambda<2>
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nk}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam1,
          poly_3mm_lam2,
          poly_3mm_lam3

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nm}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam4,
          poly_3mm_lam5,
          poly_3mm_lam6

        );

        RAJA::kernel_param<EXEC_POL>(
          RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                           RAJA::RangeSegment{0, nl},
                           RAJA::RangeSegment{0, nj}),
          RAJA::make_tuple(static_cast<Real_type>(0.0)),

          poly_3mm_lam7,
          poly_3mm_lam8,
          poly_3mm_lam9

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
