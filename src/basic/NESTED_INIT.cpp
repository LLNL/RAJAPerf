//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


#define NESTED_INIT_DATA_SETUP_CPU \
  ResReal_ptr array = m_array; \
  Index_type ni = m_ni; \
  Index_type nj = m_nj; \
  Index_type nk = m_nk;


NESTED_INIT::NESTED_INIT(const RunParams& params)
  : KernelBase(rajaperf::Basic_NESTED_INIT, params)
{
  m_ni = 500;
  m_nj = 500;
  m_nk = m_nk_init = 50;

  setDefaultSize(m_ni * m_nj * m_nk);
  setDefaultReps(100);
}

NESTED_INIT::~NESTED_INIT() 
{
}

void NESTED_INIT::setUp(VariantID vid)
{
  m_nk = m_nk_init * static_cast<Real_type>( getRunSize() ) / getDefaultSize();
  m_array_length = m_ni * m_nj * m_nk;

  allocAndInitDataConst(m_array, m_array_length, 0.0, vid);
}

void NESTED_INIT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP_CPU;

  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < nk; ++k ) {
          for (Index_type j = 0; j < nj; ++j ) {
            for (Index_type i = 0; i < ni; ++i ) {
              NESTED_INIT_BODY;
            }
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
          RAJA::statement::For<2, RAJA::loop_exec,    // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::For<0, RAJA::loop_exec,// i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                 RAJA::RangeSegment(0, nj),
                                                 RAJA::RangeSegment(0, nk)),
                                nestedinit_lam
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

#if 0
// using collapse here doesn't appear to yield a performance benefit
          #pragma omp parallel for collapse(3)
#else
          #pragma omp parallel for
#endif
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                NESTED_INIT_BODY;
              }
            }
          }

      }
      stopTimer();

      break;
    }

    case OpenMP_Lambda : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#if 0
// using collapse here doesn't appear to yield a performance benefit
          #pragma omp parallel for collapse(3)
#else
          #pragma omp parallel for
#endif
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                nestedinit_lam(i, j, k);
              }
            }
          }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

#if 0
// To compare OpenMP collapse with RAJA-based OpenMP, use this policy
      using EXEC_POL =
        RAJA::KernelPolicy<
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    RAJA::ArgList<2, 1, 0>,  // k, j, i
            RAJA::statement::Lambda<0>
          >
        >;
#else
      using EXEC_POL = 
        RAJA::KernelPolicy<
          RAJA::statement::For<2, RAJA::omp_parallel_for_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,            // j
              RAJA::statement::For<0, RAJA::loop_exec,          // i
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;
#endif

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                                 RAJA::RangeSegment(0, nj),
                                                 RAJA::RangeSegment(0, nk)),
                                nestedinit_lam
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
      std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void NESTED_INIT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_array, m_array_length);
}

void NESTED_INIT::tearDown(VariantID vid)
{
  (void) vid;
  RAJA::free_aligned(m_array);
  m_array = 0; 
}

} // end namespace basic
} // end namespace rajaperf
