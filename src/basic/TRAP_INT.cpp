/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel TRAP_INT.
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


#include "TRAP_INT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>
#if 0
#include <iomanip>
#endif

namespace rajaperf 
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_HOST_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


#define TRAP_INT_DATA \
  Real_type x0 = m_x0; \
  Real_type xp = m_xp; \
  Real_type y = m_y; \
  Real_type yp = m_yp; \
  Real_type h = m_h;

#define TRAP_INT_BODY \
  Real_type x = x0 + i*h; \
  sumx += trap_int_func(x, y, xp, yp);


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define TRAP_INT_DATA_SETUP_CUDA // nothing to do here...

#define TRAP_INT_DATA_TEARDOWN_CUDA // nothing to do here...

#if 0
__global__ void trapint(Real_type x0, Real_type xp,
                        Real_type y, Real_type yp, 
                        Real_type h, 
                        Real_ptr sumx,
                        Index_type iend)
{
   extern __shared__ Real_type tsumx[];

   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     Index_type tid = threadIdx.x;
     Real_type x = x0 + i*h;
     Real_type tsumx = trap_int_func(x, y, xp, yp);

     tsumx[tid] = tsumx;
     __syncthreads();

     for (Index_type s = blockDim.x/2; s > 0; s /= 2) {
       if (tid < s) tsumx[tid] += tsumx[tid+s];
       __syncthreads();
     }

     if (tid == 0) atomcAdd(*sumx, tsumx[tid]);  // Need "atomicAdd for reals
   }
}
#endif

#endif // if defined(RAJA_ENABLE_CUDA)


TRAP_INT::TRAP_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_TRAP_INT, params)
{
   setDefaultSize(100000);
   setDefaultSamples(1200);
}

TRAP_INT::~TRAP_INT() 
{
}

void TRAP_INT::setUp(VariantID vid)
{
  Real_type xn; 
  initData(xn, vid);

  initData(m_x0, vid);
  initData(m_xp, vid);
  initData(m_y,  vid);
  initData(m_yp, vid);

  m_h = xn - m_x0;

  m_sumx_init = 0.5*( trap_int_func(m_x0, m_y, m_xp, m_yp) +
                      trap_int_func(xn, m_y, m_xp, m_yp) );  

  m_sumx = 0;
}

void TRAP_INT::runKernel(VariantID vid)
{
#if 0
std::cout << "\tTRAP(" << vid << ") : sumx = " 
          << std::setprecision(20) << m_sumx_init << std::endl; 
#endif
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      TRAP_INT_DATA;

      Real_type sumx = m_sumx_init;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

      }

      m_sumx = sumx * h;

      stopTimer();

      break;
    }

    case RAJA_Seq : {

      TRAP_INT_DATA;

      RAJA::ReduceSum<RAJA::seq_reduce, Real_type> sumx(m_sumx_init);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::seq_exec>(ibegin, iend, [=](int i) {
          TRAP_INT_BODY;
        });

      }

      m_sumx = static_cast<Real_type>(sumx.get()) * h;

      stopTimer();

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      TRAP_INT_DATA;

      Real_type sumx = m_sumx_init;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for reduction(+:sumx)
        for (Index_type i = ibegin; i < iend; ++i ) {
          TRAP_INT_BODY;
        }

      }

      m_sumx = sumx * h;

      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      TRAP_INT_DATA;

      RAJA::ReduceSum<RAJA::omp_reduce, Real_type> sumx(m_sumx_init);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          TRAP_INT_BODY;
        });

      }

      m_sumx = static_cast<Real_type>(sumx.get()) * h;

      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

#if 0
      TRAP_INT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         trapint<<<grid_size, block_size>>>( iend );

      }
      stopTimer();

      TRAP_INT_DATA_TEARDOWN_CUDA;
#endif

      break;
    }

    case RAJA_CUDA : {

      TRAP_INT_DATA;

      RAJA::ReduceSum<RAJA::cuda_reduce<block_size>, Real_type> sumx(m_sumx_init);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           TRAP_INT_BODY;
         });

      }

      m_sumx = static_cast<Real_type>(sumx.get()) * h;

      stopTimer();

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

#if 0
std::cout << "\t\t sumx = "
          << std::setprecision(20) << m_sumx << std::endl; 
#endif
}

void TRAP_INT::updateChecksum(VariantID vid)
{
#if 1
  checksum[vid] += (m_sumx + 0.00123) / (m_sumx - 0.00123);
#else
  checksum[vid] += m_sumx;
#endif
}

void TRAP_INT::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
