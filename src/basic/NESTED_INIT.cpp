/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel NESTED_INIT.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "NESTED_INIT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

#include <iostream>

//#define USE_COLLAPSE
#undef USE_COLLAPSE


namespace rajaperf 
{
namespace basic
{

#define NESTED_INIT_DATA \
  ResReal_ptr array = m_array; \
  Int_type ni = m_ni; \
  Int_type nj = m_nj; \
  Int_type nk = m_nk;

#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;


#if defined(RAJA_ENABLE_CUDA)

#define NESTED_INIT_DATA_SETUP_CUDA \
  Real_ptr array; \
  Int_type ni = m_ni; \
  Int_type nj = m_nj; \
  Int_type nk = m_nk; \
\
  allocAndInitCudaDeviceData(array, m_array, ni * nj * nk);

#define NESTED_INIT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_array, array, ni * nj * nk); \
  deallocCudaDeviceData(array);

__global__ void nested_init(Real_ptr array,
                            Int_type ni, Int_type nj)
{
   Index_type i = threadIdx.x;
   Index_type j = blockIdx.y;
   Index_type k = blockIdx.z;

   NESTED_INIT_BODY; 
}

#endif // if defined(RAJA_ENABLE_CUDA)


NESTED_INIT::NESTED_INIT(const RunParams& params)
  : KernelBase(rajaperf::Basic_NESTED_INIT, params)
{
  m_ni = 500;
  m_nj = 500;
  m_nk = m_nk_init = 50;

  setDefaultSize(m_ni * m_nj * m_nk);
  setDefaultSamples(100);
}

NESTED_INIT::~NESTED_INIT() 
{
}

void NESTED_INIT::setUp(VariantID vid)
{
  (void) vid;

  m_nk = m_nk_init * static_cast<Real_type>( getRunSize() ) / getDefaultSize();

  int len = m_ni * m_nj * m_nk;
  m_array = RAJA::allocate_aligned_type<Real_type>(RAJA::DATA_ALIGN,
                                                   len*sizeof(Real_type)); 
}

void NESTED_INIT::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();

  switch ( vid ) {

    case Baseline_Seq : {

      NESTED_INIT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

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

    case RAJA_Seq : {

      NESTED_INIT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::simd_exec,
                                       RAJA::seq_exec,
                                       RAJA::seq_exec >, 
                       RAJA::Permute<RAJA::PERM_KJI> > >(
              RAJA::RangeSegment(0, ni),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, nk),
          [=](Index_type i, Index_type j, Index_type k) {     
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      NESTED_INIT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

#if defined(USE_COLLAPSE)
        #pragma omp parallel 
          {
            #pragma omp for nowait collapse(3) 
            for (Index_type k = 0; k < nk; ++k ) {
              for (Index_type j = 0; j < nj; ++j ) {
                for (Index_type i = 0; i < ni; ++i ) {
                  NESTED_INIT_BODY;
                }
              }
            }
          } // omp parallel
#else
          #pragma omp parallel for 
          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                NESTED_INIT_BODY;
              }
            }
          }
#endif

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      NESTED_INIT_DATA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

#if defined(USE_COLLAPSE) 
      // impact....is there something wrong with the forallN implementation?
        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::simd_exec,
                                       RAJA::omp_collapse_nowait_exec,
                                       RAJA::omp_collapse_nowait_exec >, 
                       RAJA::Permute<RAJA::PERM_KJI> > >(
              RAJA::RangeSegment(0, ni),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, nk),
          [=](Index_type i, Index_type j, Index_type k) {     
          NESTED_INIT_BODY;
        });
#else
        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::simd_exec,
                                       RAJA::seq_exec,
                                       RAJA::omp_parallel_for_exec >, 
                       RAJA::Permute<RAJA::PERM_KJI> > >(
              RAJA::RangeSegment(0, ni),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, nk),
          [=](Index_type i, Index_type j, Index_type k) {     
          NESTED_INIT_BODY;
        });
#endif

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

      NESTED_INIT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        dim3 nthreads_per_block(ni, 1, 1);
        dim3 nblocks(1, nj, nk);

        nested_init<<<nblocks, nthreads_per_block>>>(array,
                                                     ni, nj);

      }
      stopTimer();

      NESTED_INIT_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      NESTED_INIT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::cuda_thread_x_exec,
                                       RAJA::cuda_block_y_exec,
                                       RAJA::cuda_block_z_exec >, 
                       RAJA::Permute<RAJA::PERM_KJI> > >(
              RAJA::RangeSegment(0, ni),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, nk),
          [=] __device__ (Index_type i, Index_type j, Index_type k) {     
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

      NESTED_INIT_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

#if 0
    case Baseline_OpenMP4x :
    case RAJA_OpenMP4x : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void NESTED_INIT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_array, m_ni * m_nj * m_nk);
}

void NESTED_INIT::tearDown(VariantID vid)
{
  (void) vid;
  RAJA::free_aligned(m_array);
  m_array = 0; 
}

} // end namespace basic
} // end namespace rajaperf
