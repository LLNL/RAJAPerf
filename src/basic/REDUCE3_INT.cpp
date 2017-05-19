/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel REDUCE3_INT.
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


#include "REDUCE3_INT.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <limits>
#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define REDUCE3_INT_DATA \
  Int_ptr vec = m_vec; \

#define REDUCE3_INT_BODY  \
  vsum += vec[i] ; \
  vmin = RAJA_MIN(vmin, vec[i]) ; \
  vmax = RAJA_MAX(vmax, vec[i]) ;

#define REDUCE3_INT_BODY_RAJA  \
  vsum += vec[i] ; \
  vmin.min(vec[i]) ; \
  vmax.max(vec[i]) ;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define REDUCE3_INT_DATA_SETUP_CUDA \
  Int_ptr vec; \
\
  allocAndInitCudaDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(vec);

#if 0
__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_ptr vmin, Int_ptr vmax,
                           Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     REDUCE3_INT_BODY; 
   }
}
#endif

#endif // if defined(RAJA_ENABLE_CUDA)


REDUCE3_INT::REDUCE3_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_REDUCE3_INT, params)
{
   setDefaultSize(1000000);
   setDefaultSamples(1000);
}

REDUCE3_INT::~REDUCE3_INT() 
{
}

void REDUCE3_INT::setUp(VariantID vid)
{
  allocAndInitData(m_vec, getRunSize(), vid);

  m_vsum = 0;
  m_vmin = std::numeric_limits<Int_type>::max();
  m_vmax = std::numeric_limits<Int_type>::min();
}

void REDUCE3_INT::runKernel(VariantID vid)
{
  const Index_type run_samples = getRunSamples();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Baseline_Seq : {

      REDUCE3_INT_DATA;

      Int_type vsum = m_vsum;
      Int_type vmin = m_vmin;
      Int_type vmax = m_vmax;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

      }
      stopTimer();

      m_vsum = vsum;
      m_vmin = vmin;
      m_vmax = vmax;

      break;
    }

    case RAJA_Seq : {

      REDUCE3_INT_DATA;

      RAJA::ReduceSum<RAJA::seq_reduce, Int_type> vsum(m_vsum);
      RAJA::ReduceMin<RAJA::seq_reduce, Int_type> vmin(m_vmin);
      RAJA::ReduceMax<RAJA::seq_reduce, Int_type> vmax(m_vmax);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          REDUCE3_INT_BODY_RAJA;
        });

      }
      stopTimer();

      m_vsum = static_cast<Real_type>(vsum.get());
      m_vmin = static_cast<Real_type>(vmin.get());
      m_vmax = static_cast<Real_type>(vmax.get());

      break;
    }

#if defined(_OPENMP)
    case Baseline_OpenMP : {

      REDUCE3_INT_DATA;

      Int_type vsum = m_vsum;
      Int_type vmin = m_vmin;
      Int_type vmax = m_vmax;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        #pragma omp parallel for reduction(+:vsum), \
                                 reduction(min:vmin), \
                                 reduction(max:vmax)
        for (Index_type i = ibegin; i < iend; ++i ) {
          REDUCE3_INT_BODY;
        }

      }
      stopTimer();

      m_vsum = vsum;
      m_vmin = vmin;
      m_vmax = vmax;

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      REDUCE3_INT_DATA;

      RAJA::ReduceSum<RAJA::omp_reduce, Int_type> vsum(m_vsum);
      RAJA::ReduceMin<RAJA::omp_reduce, Int_type> vmin(m_vmin);
      RAJA::ReduceMax<RAJA::omp_reduce, Int_type> vmax(m_vmax);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          REDUCE3_INT_BODY_RAJA;
        });

      }
      stopTimer();

      m_vsum = static_cast<Real_type>(vsum.get());
      m_vmin = static_cast<Real_type>(vmin.get());
      m_vmax = static_cast<Real_type>(vmax.get());

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Baseline_CUDA : {

#if 0
      REDUCE3_INT_DATA_SETUP_CUDA;

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         muladdsub<<<grid_size, block_size>>>( out1, out2, out3, in1, in2, 
                                               iend ); 

      }
      stopTimer();

      REDUCE3_INT_DATA_TEARDOWN_CUDA;
#endif

      break; 
    }

    case RAJA_CUDA : {

      REDUCE3_INT_DATA_SETUP_CUDA;

      RAJA::ReduceSum<RAJA::cuda_reduce<block_size>, Int_type> vsum(m_vsum);
      RAJA::ReduceMin<RAJA::cuda_reduce<block_size>, Int_type> vmin(m_vmin);
      RAJA::ReduceMax<RAJA::cuda_reduce<block_size>, Int_type> vmax(m_vmax);

      startTimer();
      for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           REDUCE3_INT_BODY_RAJA;
         });

      }
      stopTimer();

      m_vsum = static_cast<Real_type>(vsum.get());
      m_vmin = static_cast<Real_type>(vmin.get());
      m_vmax = static_cast<Real_type>(vmax.get());

      REDUCE3_INT_DATA_TEARDOWN_CUDA;

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

void REDUCE3_INT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_vsum;
  checksum[vid] += m_vmin;
  checksum[vid] += m_vmax;
}

void REDUCE3_INT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_vec);
}

} // end namespace basic
} // end namespace rajaperf
