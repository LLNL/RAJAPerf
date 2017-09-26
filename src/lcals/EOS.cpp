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


#include "EOS.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define EOS_DATA \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr z = m_z; \
  ResReal_ptr u = m_u; \
\
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t;

#define EOS_BODY  \
  x[i] = u[i] + r*( z[i] + r*y[i] ) + \
                t*( u[i+3] + r*( u[i+2] + r*u[i+1] ) + \
                   t*( u[i+6] + q*( u[i+5] + q*u[i+4] ) ) );


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define EOS_DATA_SETUP_CUDA \
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr z; \
  Real_ptr u; \
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t; \
\
  allocAndInitCudaDeviceData(x, m_x, iend+7); \
  allocAndInitCudaDeviceData(y, m_y, iend+7); \
  allocAndInitCudaDeviceData(z, m_z, iend+7); \
  allocAndInitCudaDeviceData(u, m_u, iend+7);

#define EOS_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \
  deallocCudaDeviceData(u);

__global__ void eos(Real_ptr x, Real_ptr y, Real_ptr z, Real_ptr u,
                    Real_type q, Real_type r, Real_type t,
                    Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     EOS_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


EOS::EOS(const RunParams& params)
  : KernelBase(rajaperf::Lcals_EOS, params)
{
   setDefaultSize(100000);
   setDefaultReps(5000);
}

EOS::~EOS() 
{
}

void EOS::setUp(VariantID vid)
{
  allocAndInitData(m_x, getRunSize()+7, vid);
  allocAndInitData(m_y, getRunSize()+7, vid);
  allocAndInitData(m_z, getRunSize()+7, vid);
  allocAndInitData(m_u, getRunSize()+7, vid);

  initData(m_q, vid);
  initData(m_r, vid);
  initData(m_t, vid);
}

void EOS::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      EOS_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          EOS_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      EOS_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](Index_type i) {
          EOS_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      EOS_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          EOS_BODY;
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

      EOS_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, 
          [=](Index_type i) {
          EOS_BODY;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      EOS_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         eos<<<grid_size, block_size>>>( x, y, z, u, 
                                         q, r, t,
                                         iend ); 

      }
      stopTimer();

      EOS_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      EOS_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend, 
           [=] __device__ (Index_type i) {
           EOS_BODY;
         });

      }
      stopTimer();

      EOS_DATA_TEARDOWN_CUDA;

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

void EOS::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void EOS::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
  deallocData(m_u);
}

} // end namespace lcals
} // end namespace rajaperf
