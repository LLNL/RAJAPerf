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


#include "FIRST_DIFF.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define FIRST_DIFF_DATA \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y;

#define FIRST_DIFF_BODY  \
  x[i] = y[i+1] - y[i];


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define FIRST_DIFF_DATA_SETUP_CUDA \
  Real_ptr x; \
  Real_ptr y; \
\
  allocAndInitCudaDeviceData(x, m_x, iend+1); \
  allocAndInitCudaDeviceData(y, m_y, iend+1);

#define FIRST_DIFF_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);

__global__ void first_diff(Real_ptr x, Real_ptr y,
                           Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIRST_DIFF_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


FIRST_DIFF::FIRST_DIFF(const RunParams& params)
  : KernelBase(rajaperf::Lcals_FIRST_DIFF, params)
{
   setDefaultSize(100000);
   setDefaultReps(16000);
}

FIRST_DIFF::~FIRST_DIFF() 
{
}

void FIRST_DIFF::setUp(VariantID vid)
{
  allocAndInitData(m_x, getRunSize()+1, vid);
  allocAndInitData(m_y, getRunSize()+1, vid);
}

void FIRST_DIFF::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      FIRST_DIFF_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_DIFF_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      FIRST_DIFF_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          FIRST_DIFF_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      FIRST_DIFF_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_DIFF_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      FIRST_DIFF_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          FIRST_DIFF_BODY;
        });

      }
      stopTimer();

      break;
    }


#if defined(RAJA_ENABLE_TARGET_OPENMP)
#define NUMTEAMS 128
    case Base_OpenMPTarget : {
      FIRST_DIFF_DATA;
                       
      Index_type n = iend + 1;
      #pragma omp target enter data map(to:x[0:n],y[0:n])
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        
        for (Index_type i = ibegin; i < iend; ++i ) {
          FIRST_DIFF_BODY;
        }
      }
      stopTimer();
      #pragma omp target exit data map(from:x[0:n]) map(delete:y[0:n])
      break;
    }

    case RAJA_OpenMPTarget: {
      FIRST_DIFF_DATA;
                       
      Index_type n = iend + 1;
      #pragma omp target enter data map(to:x[0:n],y[0:n])
      startTimer();
      #pragma omp target data use_device_ptr(x,y)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          FIRST_DIFF_BODY;
        });

      }
      stopTimer();
      #pragma omp target exit data map(from:x[0:n]) map(delete:y[0:n])
      break;                        
    }                          
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      FIRST_DIFF_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         first_diff<<<grid_size, block_size>>>( x, y,
                                                iend ); 

      }
      stopTimer();

      FIRST_DIFF_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      FIRST_DIFF_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
           FIRST_DIFF_BODY;
         });

      }
      stopTimer();

      FIRST_DIFF_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void FIRST_DIFF::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void FIRST_DIFF::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
}

} // end namespace lcals
} // end namespace rajaperf
