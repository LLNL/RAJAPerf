/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Lcals kernel FIRST_DIFF.
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
// For more information, please see the file LICENSE in the top-level directory.
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


#if defined(ENABLE_CUDA)

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

#endif // if defined(ENABLE_CUDA)


FIRST_DIFF::FIRST_DIFF(const RunParams& params)
  : KernelBase(rajaperf::Lcals_FIRST_DIFF, params)
{
   setDefaultSize(100000);
   setDefaultReps(15000);
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

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          FIRST_DIFF_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(ENABLE_OPENMP)
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

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      FIRST_DIFF_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          FIRST_DIFF_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(ENABLE_CUDA)
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
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           FIRST_DIFF_BODY;
         });

      }
      stopTimer();

      FIRST_DIFF_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

#if 0
    case Base_OpenMP4x :
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
