/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel FIR.
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


#include "FIR.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define USE_CONSTANT_MEMORY
//#undef USE_CONSTANT_MEMORY

#define COEFFLEN (16)

#define FIR_COEFF \
  Real_type coeff_array[COEFFLEN] = { 3.0, -1.0, -1.0, -1.0, \
                                      -1.0, 3.0, -1.0, -1.0, \
                                      -1.0, -1.0, 3.0, -1.0, \
                                      -1.0, -1.0, -1.0, 3.0 };


#define FIR_DATA \
  ResReal_ptr in = m_in; \
  ResReal_ptr out = m_out; \
\
  Real_type coeff[COEFFLEN]; \
  std::copy(std::begin(coeff_array), std::end(coeff_array), std::begin(coeff));\
\
  const Index_type coefflen = m_coefflen;


#define FIR_BODY \
  Real_type sum = 0.0; \
\
  for (Index_type j = 0; j < coefflen; ++j ) { \
    sum += coeff[j]*in[i+j]; \
  } \
  out[i] = sum;


#if defined(ENABLE_CUDA)
  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#if defined(USE_CONSTANT_MEMORY)

__constant__ Real_type coeff[COEFFLEN];

#define FIR_DATA_SETUP_CUDA \
  Real_ptr in; \
  Real_ptr out; \
\
  const Index_type coefflen = m_coefflen; \
\
  allocAndInitCudaDeviceData(in, m_in, getRunSize()); \
  allocAndInitCudaDeviceData(out, m_out, getRunSize()); \
  cudaMemcpyToSymbol(coeff, coeff_array, COEFFLEN * sizeof(Real_type));


#define FIR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out, out, getRunSize()); \
  deallocCudaDeviceData(in); \
  deallocCudaDeviceData(out);

__global__ void fir(Real_ptr out, Real_ptr in,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#else  // use global memry for coefficients 

#define FIR_DATA_SETUP_CUDA \
  Real_ptr in; \
  Real_ptr out; \
  Real_ptr coeff; \
\
  const Index_type coefflen = m_coefflen; \
\
  allocAndInitCudaDeviceData(in, m_in, getRunSize()); \
  allocAndInitCudaDeviceData(out, m_out, getRunSize()); \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocAndInitCudaDeviceData(coeff, tcoeff, COEFFLEN);


#define FIR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_out, out, getRunSize()); \
  deallocCudaDeviceData(in); \
  deallocCudaDeviceData(out); \
  deallocCudaDeviceData(coeff);

__global__ void fir(Real_ptr out, Real_ptr in,
                    Real_ptr coeff,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#endif 

#endif // if defined(ENABLE_CUDA)


FIR::FIR(const RunParams& params)
  : KernelBase(rajaperf::Apps_FIR, params)
{
  setDefaultSize(100000);
  setDefaultReps(5000);

  m_coefflen = COEFFLEN;
}

FIR::~FIR() 
{
}

Index_type FIR::getItsPerRep() const { 
  return getRunSize() - m_coefflen;
}

void FIR::setUp(VariantID vid)
{
  allocAndInitData(m_in, getRunSize(), vid);
  allocAndInitData(m_out, getRunSize(), vid);
}

void FIR::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  switch ( vid ) {

    case Base_Seq : {

      FIR_COEFF;

      FIR_DATA;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIR_BODY;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      FIR_COEFF;

      FIR_DATA;
 
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          FIR_BODY;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(ENABLE_OPENMP)      
    case Base_OpenMP : {

      FIR_COEFF;

      FIR_DATA;
 
      startTimer();
      for (RepIndex_type irep = ibegin; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
           FIR_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // Not applicable...
      break;
    }

    case RAJA_OpenMP : {

      FIR_COEFF;

      FIR_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          FIR_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(ENABLE_CUDA)
    case Base_CUDA : {

      FIR_COEFF;

      FIR_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

#if defined(USE_CONSTANT_MEMORY)
         fir<<<grid_size, block_size>>>( out, in,
                                         coefflen,
                                         iend );
#else
         fir<<<grid_size, block_size>>>( out, in,
                                         coeff,
                                         coefflen,
                                         iend );
#endif

      }
      stopTimer();

      FIR_DATA_TEARDOWN_CUDA;

      break;
    }

    case RAJA_CUDA : {

      FIR_COEFF;

      FIR_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           FIR_BODY;
         });

      }
      stopTimer();

      FIR_DATA_TEARDOWN_CUDA;

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

void FIR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out, getRunSize());
}

void FIR::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_in);
  deallocData(m_out);
}

} // end namespace apps
} // end namespace rajaperf
