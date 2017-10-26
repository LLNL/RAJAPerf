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


#include "INT_PREDICT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define INT_PREDICT_DATA \
  ResReal_ptr px = m_px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \
  const Index_type offset = m_offset;

#define INT_PREDICT_BODY  \
  px[i] = dm28*px[i + offset * 12] + dm27*px[i + offset * 11] + \
          dm26*px[i + offset * 10] + dm25*px[i + offset *  9] + \
          dm24*px[i + offset *  8] + dm23*px[i + offset *  7] + \
          dm22*px[i + offset *  6] + \
          c0*( px[i + offset *  4] + px[i + offset *  5] ) + \
          px[i + offset *  2]; 


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INT_PREDICT_DATA_SETUP_CUDA \
  Real_ptr px; \
  Real_type dm22 = m_dm22; \
  Real_type dm23 = m_dm23; \
  Real_type dm24 = m_dm24; \
  Real_type dm25 = m_dm25; \
  Real_type dm26 = m_dm26; \
  Real_type dm27 = m_dm27; \
  Real_type dm28 = m_dm28; \
  Real_type c0 = m_c0; \
  const Index_type offset = m_offset; \
\
  allocAndInitCudaDeviceData(px, m_px, m_offset*13);

#define INT_PREDICT_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_px, px, m_offset*13); \
  deallocCudaDeviceData(px);

__global__ void int_predict(Real_ptr px,
                            Real_type dm22, Real_type dm23, Real_type dm24,
                            Real_type dm25, Real_type dm26, Real_type dm27,
                            Real_type dm28, Real_type c0,
                            const Index_type offset, 
                            Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INT_PREDICT_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
   setDefaultSize(100000);
   setDefaultReps(4000);
}

INT_PREDICT::~INT_PREDICT() 
{
}

void INT_PREDICT::setUp(VariantID vid)
{
  allocAndInitData(m_px, getRunSize()*13, vid);

  initData(m_dm22);
  initData(m_dm23);
  initData(m_dm24);
  initData(m_dm25);
  initData(m_dm26);
  initData(m_dm27);
  initData(m_dm28);
  initData(m_c0);

  m_offset = getRunSize();
}

void INT_PREDICT::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      INT_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INT_PREDICT_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INT_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) { 
          INT_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      INT_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INT_PREDICT_BODY;
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

      INT_PREDICT_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INT_PREDICT_BODY;
        });

      }
      stopTimer();

      break;
    }


#if defined(RAJA_ENABLE_TARGET_OPENMP)
#define NUMTEAMS 128
    case Base_OpenMPTarget : {
      INT_PREDICT_DATA;
                              
      Index_type n = m_offset*13;
      
      #pragma omp target enter data map(to:px[0:n],dm22,dm23,dm24,dm25,dm26,dm27,dm28,c0,offset)
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        
        for (Index_type i = ibegin; i < iend; ++i ) {
          INT_PREDICT_BODY;
        }
      }
      stopTimer();
      #pragma omp target exit data map(from:px[0:n]) map(delete:dm22,dm23,dm24,dm25,dm26,dm27,dm28,c0,offset)
      break;
    }

    case RAJA_OpenMPTarget: {
      INT_PREDICT_DATA;
                              
      Index_type n = m_offset*13;
      
      #pragma omp target enter data map(to:px[0:n],dm22,dm23,dm24,dm25,dm26,dm27,dm28,c0,offset)
      startTimer();
      #pragma omp target data use_device_ptr(px)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(ibegin, iend, 
          [=](Index_type i) {
          INT_PREDICT_BODY;
        });

      }
      stopTimer();
      #pragma omp target exit data map(from:px[0:n]) map(delete:dm22,dm23,dm24,dm25,dm26,dm27,dm28,c0,offset)
      break;
    }
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             


#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      INT_PREDICT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         int_predict<<<grid_size, block_size>>>( px, 
                                                 dm22, dm23, dm24, dm25,
                                                 dm26, dm27, dm28, c0,
                                                 offset,
                                                 iend ); 

      }
      stopTimer();

      INT_PREDICT_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      INT_PREDICT_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
           INT_PREDICT_BODY;
         });

      }
      stopTimer();

      INT_PREDICT_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

}

void INT_PREDICT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_px, m_offset*13);
}

void INT_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
}

} // end namespace lcals
} // end namespace rajaperf
