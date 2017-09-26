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
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "DIFF_PREDICT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define DIFF_PREDICT_DATA \
  ResReal_ptr px = m_px; \
  ResReal_ptr cx = m_cx; \
  const Index_type offset = m_offset;

#define DIFF_PREDICT_BODY  \
  Real_type ar, br, cr; \
\
  ar                  = cx[i + offset * 4];       \
  br                  = ar - px[i + offset * 4];  \
  px[i + offset * 4]  = ar;                       \
  cr                  = br - px[i + offset * 5];  \
  px[i + offset * 5]  = br;                       \
  ar                  = cr - px[i + offset * 6];  \
  px[i + offset * 6]  = cr;                       \
  br                  = ar - px[i + offset * 7];  \
  px[i + offset * 7]  = ar;                       \
  cr                  = br - px[i + offset * 8];  \
  px[i + offset * 8]  = br;                       \
  ar                  = cr - px[i + offset * 9];  \
  px[i + offset * 9]  = cr;                       \
  br                  = ar - px[i + offset * 10]; \
  px[i + offset * 10] = ar;                       \
  cr                  = br - px[i + offset * 11]; \
  px[i + offset * 11] = br;                       \
  px[i + offset * 13] = cr - px[i + offset * 12]; \
  px[i + offset * 12] = cr;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define DIFF_PREDICT_DATA_SETUP_CUDA \
  Real_ptr px; \
  Real_ptr cx; \
  const Index_type offset = m_offset; \
\
  allocAndInitCudaDeviceData(px, m_px, m_offset*14); \
  allocAndInitCudaDeviceData(cx, m_cx, m_offset*14);

#define DIFF_PREDICT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_px, px, m_offset*14); \
  deallocCudaDeviceData(px); \
  deallocCudaDeviceData(cx);

__global__ void diff_predict(Real_ptr px, Real_ptr cx,
                             const Index_type offset, 
                             Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     DIFF_PREDICT_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


DIFF_PREDICT::DIFF_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_DIFF_PREDICT, params)
{
   setDefaultSize(100000);
   setDefaultReps(2000);
}

DIFF_PREDICT::~DIFF_PREDICT() 
{
}

void DIFF_PREDICT::setUp(VariantID vid)
{
  allocAndInitData(m_px, getRunSize()*14, vid);
  allocAndInitData(m_cx, getRunSize()*14, vid);

  m_offset = getRunSize();
}

void DIFF_PREDICT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      DIFF_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          DIFF_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      DIFF_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          DIFF_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      DIFF_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          DIFF_PREDICT_BODY;
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

      DIFF_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          DIFF_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      DIFF_PREDICT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         diff_predict<<<grid_size, block_size>>>( px, cx,
                                                  offset,
                                                  iend ); 

      }
      stopTimer();

      DIFF_PREDICT_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      DIFF_PREDICT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           DIFF_PREDICT_BODY;
         });

      }
      stopTimer();

      DIFF_PREDICT_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

#if 0
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void DIFF_PREDICT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_px, m_offset*14);
}

void DIFF_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
  deallocData(m_cx);
}

} // end namespace lcals
} // end namespace rajaperf
