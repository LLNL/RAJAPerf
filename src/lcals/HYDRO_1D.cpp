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


#include "HYDRO_1D.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define HYDRO_1D_DATA \
  ResReal_ptr x = m_x; \
  ResReal_ptr y = m_y; \
  ResReal_ptr z = m_z; \
\
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t;

#define HYDRO_1D_BODY  \
  x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define HYDRO_1D_DATA_SETUP_CUDA \
  Real_ptr x; \
  Real_ptr y; \
  Real_ptr z; \
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t; \
\
  allocAndInitCudaDeviceData(x, m_x, iend+12); \
  allocAndInitCudaDeviceData(y, m_y, iend+12); \
  allocAndInitCudaDeviceData(z, m_z, iend+12);

#define HYDRO_1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \

__global__ void hydro_1d(Real_ptr x, Real_ptr y, Real_ptr z,
                         Real_type q, Real_type r, Real_type t,
                         Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     HYDRO_1D_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


HYDRO_1D::HYDRO_1D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_1D, params)
{
   setDefaultSize(100000);
   setDefaultReps(12500);
}

HYDRO_1D::~HYDRO_1D() 
{
}

void HYDRO_1D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_x, getRunSize()+12, 0.0, vid);
  allocAndInitData(m_y, getRunSize()+12, vid);
  allocAndInitData(m_z, getRunSize()+12, vid);

  initData(m_q, vid);
  initData(m_r, vid);
  initData(m_t, vid);
}

void HYDRO_1D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      HYDRO_1D_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      HYDRO_1D_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          HYDRO_1D_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      HYDRO_1D_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      HYDRO_1D_DATA;
      
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          HYDRO_1D_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#define NUMTEAMS 128

    case Base_OpenMPTarget : {

      HYDRO_1D_DATA;

      Index_type n = iend + 12;
      #pragma omp target enter data map(to:x[0:n],y[0:n],z[0:n],q,r,t)

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }

      }
      stopTimer();

      #pragma omp target exit data map(from:x[0:n]) map(delete:y[0:n],z[0:n],q,r,t)

      break;
    }

    case RAJA_OpenMPTarget: {
                              
      HYDRO_1D_DATA;

      Index_type n = iend + 12;
      #pragma omp target enter data map(to:x[0:n],y[0:n],z[0:n],q,r,t)

      startTimer();
      #pragma omp target data use_device_ptr(x,y,z)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
            RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          HYDRO_1D_BODY;
        });

      }
      stopTimer();

      #pragma omp target exit data map(from:x[0:n]) map(delete:y[0:n],z[0:n],q,r,t)

      break;                        
    }                          
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             


#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      HYDRO_1D_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         hydro_1d<<<grid_size, block_size>>>( x, y, z,
                                              q, r, t,
                                              iend ); 

      }
      stopTimer();

      HYDRO_1D_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      HYDRO_1D_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
           HYDRO_1D_BODY;
         });

      }
      stopTimer();

      HYDRO_1D_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

    default : {
      std::cout << "\n  HYDRO_1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void HYDRO_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void HYDRO_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace lcals
} // end namespace rajaperf
