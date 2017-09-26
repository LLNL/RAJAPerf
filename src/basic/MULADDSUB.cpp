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


#include "MULADDSUB.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define MULADDSUB_DATA \
  ResReal_ptr out1 = m_out1; \
  ResReal_ptr out2 = m_out2; \
  ResReal_ptr out3 = m_out3; \
  ResReal_ptr in1 = m_in1; \
  ResReal_ptr in2 = m_in2;

#define MULADDSUB_BODY  \
  out1[i] = in1[i] * in2[i] ; \
  out2[i] = in1[i] + in2[i] ; \
  out3[i] = in1[i] - in2[i] ;


#if defined(ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define MULADDSUB_DATA_SETUP_CUDA \
  Real_ptr out1; \
  Real_ptr out2; \
  Real_ptr out3; \
  Real_ptr in1; \
  Real_ptr in2; \
\
  allocAndInitCudaDeviceData(out1, m_out1, iend); \
  allocAndInitCudaDeviceData(out2, m_out2, iend); \
  allocAndInitCudaDeviceData(out3, m_out3, iend); \
  allocAndInitCudaDeviceData(in1, m_in1, iend); \
  allocAndInitCudaDeviceData(in2, m_in2, iend);

#define MULADDSUB_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out1, out1, iend); \
  getCudaDeviceData(m_out2, out2, iend); \
  getCudaDeviceData(m_out3, out3, iend); \
  deallocCudaDeviceData(out1); \
  deallocCudaDeviceData(out2); \
  deallocCudaDeviceData(out3); \
  deallocCudaDeviceData(in1); \
  deallocCudaDeviceData(in2);

__global__ void muladdsub(Real_ptr out1, Real_ptr out2, Real_ptr out3, 
                          Real_ptr in1, Real_ptr in2, 
                          Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     MULADDSUB_BODY; 
   }
}

#endif // if defined(ENABLE_CUDA)


MULADDSUB::MULADDSUB(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULADDSUB, params)
{
   setDefaultSize(100000);
   setDefaultReps(3500);
}

MULADDSUB::~MULADDSUB() 
{
}

void MULADDSUB::setUp(VariantID vid)
{
  allocAndInitData(m_out1, getRunSize(), vid);
  allocAndInitData(m_out2, getRunSize(), vid);
  allocAndInitData(m_out3, getRunSize(), vid);
  allocAndInitData(m_in1, getRunSize(), vid);
  allocAndInitData(m_in2, getRunSize(), vid);
}

void MULADDSUB::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      MULADDSUB_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          MULADDSUB_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      MULADDSUB_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          MULADDSUB_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(ENABLE_OPENMP)
    case Base_OpenMP : {

      MULADDSUB_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          MULADDSUB_BODY;
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

      MULADDSUB_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          MULADDSUB_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(ENABLE_CUDA)
    case Base_CUDA : {

      MULADDSUB_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         muladdsub<<<grid_size, block_size>>>( out1, out2, out3, in1, in2, 
                                               iend ); 

      }
      stopTimer();

      MULADDSUB_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      MULADDSUB_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           MULADDSUB_BODY;
         });

      }
      stopTimer();

      MULADDSUB_DATA_TEARDOWN_CUDA;

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

void MULADDSUB::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunSize());
  checksum[vid] += calcChecksum(m_out2, getRunSize());
  checksum[vid] += calcChecksum(m_out3, getRunSize());
}

void MULADDSUB::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_out1);
  deallocData(m_out2);
  deallocData(m_out3);
  deallocData(m_in1);
  deallocData(m_in2);
}

} // end namespace basic
} // end namespace rajaperf
