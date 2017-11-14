//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NESTED_INIT kernel reference implementation:
///
/// for (Index_type k = 0; k < nk; ++k ) {
///   for (Index_type j = 0; j < nj; ++j ) {
///     for (Index_type i = 0; i < ni; ++i ) {
///       array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
///     }
///   }
/// }
///

#include "NESTED_INIT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

#include <iostream>

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
  setDefaultReps(100);
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
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      NESTED_INIT_DATA;

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

    case RAJA_Seq : {

#if defined(USE_FORALLN_FOR_SEQ)

      NESTED_INIT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::seq_exec,
                                       RAJA::seq_exec,
                                       RAJA::simd_exec > > > (
              RAJA::RangeSegment(0, nk),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, ni),
          [=](Index_type k, Index_type j, Index_type i) {     
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

#else // use RAJA::nested

      NESTED_INIT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(RAJA::nested::Policy< 
                             RAJA::nested::For<2, RAJA::seq_exec>,      // k
                             RAJA::nested::For<1, RAJA::seq_exec>,      // j
                             RAJA::nested::For<0, RAJA::simd_exec> >{}, // i
                             camp::make_tuple(RAJA::RangeSegment(0, ni),
                                              RAJA::RangeSegment(0, nj),
                                              RAJA::RangeSegment(0, nk)),
             [=](Index_type i, Index_type j, Index_type k) {     
             NESTED_INIT_BODY;
        });

      }
      stopTimer();

#endif

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      NESTED_INIT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          #pragma omp parallel for collapse(2)
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

    case RAJA_OpenMP : {

#if defined(USE_FORALLN_FOR_OPENMP)

      NESTED_INIT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::omp_parallel_for_exec,
                                       RAJA::seq_exec,
                                       RAJA::simd_exec > > > (
              RAJA::RangeSegment(0, nk),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, ni),
          [=](Index_type k, Index_type j, Index_type i) {     
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

#else // use RAJA::nested

      NESTED_INIT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(RAJA::nested::Policy< 
                             RAJA::nested::For<2, RAJA::omp_parallel_for_exec>,      // k
                             RAJA::nested::For<1, RAJA::seq_exec>,      // j
                             RAJA::nested::For<0, RAJA::simd_exec> >{}, // i
                             camp::make_tuple(RAJA::RangeSegment(0, ni),
                                              RAJA::RangeSegment(0, nj),
                                              RAJA::RangeSegment(0, nk)),
             [=](Index_type i, Index_type j, Index_type k) {     
             NESTED_INIT_BODY;
        });

      }
      stopTimer();

#endif

      break;
    }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
#define NUMTEAMS 128
    case Base_OpenMPTarget : {

      NESTED_INIT_DATA;

      #pragma omp target enter data map(to:array[0:ni * nj * nk],ni,nj,nk)
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) collapse(3) 
        for (Index_type k = 0; k < nk; ++k ) {
          for (Index_type j = 0; j < nj; ++j ) {
            for (Index_type i = 0; i < ni; ++i ) {
              NESTED_INIT_BODY;
            }
          }
        }  
      }
      stopTimer();
      #pragma omp target exit data map(from:array[0:ni * nj * nk]) map(delete:ni,nj,nk)
      break;
    }

#if 0  // crashes clang-coral compiler      
    case RAJA_OpenMPTarget: {
                              
      NESTED_INIT_DATA;

      #pragma omp target enter data map(to:array[0:ni * nj * nk],ni,nj,nk)
      startTimer();
      #pragma omp target data use_device_ptr(array)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::simd_exec,
                                       RAJA::seq_exec,
                                       RAJA::omp_target_parallel_for_exec<NUMTEAMS>>, 
                       RAJA::Permute<RAJA::PERM_KJI> > >(
              RAJA::RangeSegment(0, ni),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, nk),
          [=](Index_type i, Index_type j, Index_type k) {     
          NESTED_INIT_BODY;
        });

      }
      stopTimer();
    
      #pragma omp target exit data map(from:array[0:ni * nj * nk]) map(delete:ni,nj,nk)
      break;                        
    }  
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP   
#endif                            

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      NESTED_INIT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

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

#if defined(USE_FORALLN_FOR_CUDA)

      NESTED_INIT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< RAJA::NestedPolicy< 
                       RAJA::ExecList< RAJA::cuda_block_z_exec,
                                       RAJA::cuda_block_y_exec,
                                       RAJA::cuda_thread_x_exec > > > (
              RAJA::RangeSegment(0, nk),
              RAJA::RangeSegment(0, nj),
              RAJA::RangeSegment(0, ni),
          [=] __device__ (Index_type k, Index_type j, Index_type i) {
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

      NESTED_INIT_DATA_TEARDOWN_CUDA;

#else // use RAJA::nested

      NESTED_INIT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(RAJA::nested::Policy<
                             RAJA::nested::For<2, RAJA::cuda_block_z_exec>, // k
                             RAJA::nested::For<1, RAJA::cuda_block_y_exec>, // j
                             RAJA::nested::For<0, RAJA::cuda_thread_x_exec> >{}, // i
                             camp::make_tuple(RAJA::RangeSegment(0, ni),
                                              RAJA::RangeSegment(0, nj),
                                              RAJA::RangeSegment(0, nk)),
          [=](Index_type i, Index_type j, Index_type k) {
          NESTED_INIT_BODY;
        });

      }
      stopTimer();

      NESTED_INIT_DATA_TEARDOWN_CUDA;

#endif

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
