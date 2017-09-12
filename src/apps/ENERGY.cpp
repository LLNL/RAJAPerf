/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel ENERGY.
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


#include "ENERGY.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

#define ENERGY_DATA \
  ResReal_ptr e_new = m_e_new; \
  ResReal_ptr e_old = m_e_old; \
  ResReal_ptr delvc = m_delvc; \
  ResReal_ptr p_new = m_p_new; \
  ResReal_ptr p_old = m_p_old; \
  ResReal_ptr q_new = m_q_new; \
  ResReal_ptr q_old = m_q_old; \
  ResReal_ptr work = m_work; \
  ResReal_ptr compHalfStep = m_compHalfStep; \
  ResReal_ptr pHalfStep = m_pHalfStep; \
  ResReal_ptr bvc = m_bvc; \
  ResReal_ptr pbvc = m_pbvc; \
  ResReal_ptr ql_old = m_ql_old; \
  ResReal_ptr qq_old = m_qq_old; \
  ResReal_ptr vnewc = m_vnewc; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut;


#define ENERGY_BODY1 \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_BODY2 \
  if ( delvc[i] > 0.0 ) { \
     q_new[i] = 0.0 ; \
  } \
  else { \
     Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ; \
     Real_type ssc = ( pbvc[i] * e_new[i] \
        + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
  }

#define ENERGY_BODY3 \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_BODY4 \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_BODY5 \
  Real_type q_tilde ; \
  if (delvc[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
         + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old[i]) ; \
  } \
  e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i]) \
                         - 8.0*(pHalfStep[i] + q_new[i]) \
                         + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ; \
  if ( fabs(e_new[i]) < e_cut ) { \
     e_new[i] = 0.0  ; \
  } \
  if ( e_new[i]  < emin ) { \
     e_new[i] = emin ; \
  }

#define ENERGY_BODY6 \
  if ( delvc[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
             + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
     if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ; \
  }

#if defined(ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define ENERGY_DATA_SETUP_CUDA \
  Real_ptr e_new; \
  Real_ptr e_old; \
  Real_ptr delvc; \
  Real_ptr p_new; \
  Real_ptr p_old; \
  Real_ptr q_new; \
  Real_ptr q_old; \
  Real_ptr work; \
  Real_ptr compHalfStep; \
  Real_ptr pHalfStep; \
  Real_ptr bvc; \
  Real_ptr pbvc; \
  Real_ptr ql_old; \
  Real_ptr qq_old; \
  Real_ptr vnewc; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut; \
\
  allocAndInitCudaDeviceData(e_new, m_e_new, iend); \
  allocAndInitCudaDeviceData(e_old, m_e_old, iend); \
  allocAndInitCudaDeviceData(delvc, m_delvc, iend); \
  allocAndInitCudaDeviceData(p_new, m_p_new, iend); \
  allocAndInitCudaDeviceData(p_old, m_p_old, iend); \
  allocAndInitCudaDeviceData(q_new, m_q_new, iend); \
  allocAndInitCudaDeviceData(q_old, m_q_old, iend); \
  allocAndInitCudaDeviceData(work, m_work, iend); \
  allocAndInitCudaDeviceData(compHalfStep, m_compHalfStep, iend); \
  allocAndInitCudaDeviceData(pHalfStep, m_pHalfStep, iend); \
  allocAndInitCudaDeviceData(bvc, m_bvc, iend); \
  allocAndInitCudaDeviceData(pbvc, m_pbvc, iend); \
  allocAndInitCudaDeviceData(ql_old, m_ql_old, iend); \
  allocAndInitCudaDeviceData(qq_old, m_qq_old, iend); \
  allocAndInitCudaDeviceData(vnewc, m_vnewc, iend);

#define ENERGY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_e_new, e_new, iend); \
  getCudaDeviceData(m_q_new, q_new, iend); \
  deallocCudaDeviceData(e_new); \
  deallocCudaDeviceData(e_old); \
  deallocCudaDeviceData(delvc); \
  deallocCudaDeviceData(p_new); \
  deallocCudaDeviceData(p_old); \
  deallocCudaDeviceData(q_new); \
  deallocCudaDeviceData(q_old); \
  deallocCudaDeviceData(work); \
  deallocCudaDeviceData(compHalfStep); \
  deallocCudaDeviceData(pHalfStep); \
  deallocCudaDeviceData(bvc); \
  deallocCudaDeviceData(pbvc); \
  deallocCudaDeviceData(ql_old); \
  deallocCudaDeviceData(qq_old); \
  deallocCudaDeviceData(vnewc);

__global__ void energycalc1(Real_ptr e_new, Real_ptr e_old, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old, Real_ptr work,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY1;
   }
}

__global__ void energycalc2(Real_ptr delvc, Real_ptr q_new,
                            Real_ptr compHalfStep, Real_ptr pHalfStep,
                            Real_ptr e_new, Real_ptr bvc, Real_ptr pbvc,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY2;
   }
}

__global__ void energycalc3(Real_ptr e_new, Real_ptr delvc,
                            Real_ptr p_old, Real_ptr q_old, 
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY3;
   }
}

__global__ void energycalc4(Real_ptr e_new, Real_ptr work,
                            Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY4;
   }
}

__global__ void energycalc5(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_ptr p_old, Real_ptr q_old,
                            Real_ptr pHalfStep, Real_ptr q_new,
                            Real_type rho0, Real_type e_cut, Real_type emin,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY5;
   }
}

__global__ void energycalc6(Real_ptr delvc,
                            Real_ptr pbvc, Real_ptr e_new, Real_ptr vnewc,
                            Real_ptr bvc, Real_ptr p_new,
                            Real_ptr q_new,
                            Real_ptr ql_old, Real_ptr qq_old,
                            Real_type rho0, Real_type q_cut,
                            Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ENERGY_BODY6;
   }
}

#endif // if defined(ENABLE_CUDA)


ENERGY::ENERGY(const RunParams& params)
  : KernelBase(rajaperf::Apps_ENERGY, params)
{
  setDefaultSize(100000);
  setDefaultReps(1300);
}

ENERGY::~ENERGY() 
{
}

void ENERGY::setUp(VariantID vid)
{
  allocAndInitData(m_e_new, getRunSize(), vid);
  allocAndInitData(m_e_old, getRunSize(), vid);
  allocAndInitData(m_delvc, getRunSize(), vid);
  allocAndInitData(m_p_new, getRunSize(), vid);
  allocAndInitData(m_p_old, getRunSize(), vid);
  allocAndInitData(m_q_new, getRunSize(), vid);
  allocAndInitData(m_q_old, getRunSize(), vid);
  allocAndInitData(m_work, getRunSize(), vid);
  allocAndInitData(m_compHalfStep, getRunSize(), vid);
  allocAndInitData(m_pHalfStep, getRunSize(), vid);
  allocAndInitData(m_bvc, getRunSize(), vid);
  allocAndInitData(m_pbvc, getRunSize(), vid);
  allocAndInitData(m_ql_old, getRunSize(), vid);
  allocAndInitData(m_qq_old, getRunSize(), vid);
  allocAndInitData(m_vnewc, getRunSize(), vid);
  
  initData(m_rho0);
  initData(m_e_cut);
  initData(m_emin);
  initData(m_q_cut);
}

void ENERGY::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      ENERGY_DATA;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }
  
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }

        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

      ENERGY_DATA;
 
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY1;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY2;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY3;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY4;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY5;
        }); 

        RAJA::forall<RAJA::simd_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY6;
        }); 

      }
      stopTimer(); 

      break;
    }

#if defined(ENABLE_OPENMP)      
    case Base_OpenMP : {

      ENERGY_DATA;
 
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
          {
            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY1;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY2;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY3;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY4;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY5;
            }

            #pragma omp for nowait schedule(static)
            for (Index_type i = ibegin; i < iend; ++i ) {
              ENERGY_BODY6;
            }
          } // omp parallel

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {

      ENERGY_DATA;
      
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    
        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY1;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY2;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY3;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY4;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY5;
        }

        #pragma omp parallel for schedule(static)
        for (Index_type i = ibegin; i < iend; ++i ) {
          ENERGY_BODY6;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      ENERGY_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY1;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY2;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY3;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY4;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY5;
        });

        RAJA::forall<RAJA::omp_parallel_for_exec>(ibegin, iend, [=](int i) {
          ENERGY_BODY6;
        });

      }
      stopTimer();
      break;
    }
#endif

#if defined(ENABLE_CUDA)
    case Base_CUDA : {
    
      ENERGY_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

         energycalc1<<<grid_size, block_size>>>( e_new, e_old, delvc,
                                                 p_old, q_old, work,
                                                 iend );

         energycalc2<<<grid_size, block_size>>>( delvc, q_new,
                                                 compHalfStep, pHalfStep,
                                                 e_new, bvc, pbvc,
                                                 ql_old, qq_old,
                                                 rho0,
                                                 iend );

         energycalc3<<<grid_size, block_size>>>( e_new, delvc,
                                                 p_old, q_old,
                                                 pHalfStep, q_new,
                                                 iend );

         energycalc4<<<grid_size, block_size>>>( e_new, work,
                                                 e_cut, emin,
                                                 iend );

         energycalc5<<<grid_size, block_size>>>( delvc,
                                                 pbvc, e_new, vnewc,
                                                 bvc, p_new,
                                                 ql_old, qq_old,
                                                 p_old, q_old,
                                                 pHalfStep, q_new,
                                                 rho0, e_cut, emin,
                                                 iend );

         energycalc6<<<grid_size, block_size>>>( delvc,
                                                 pbvc, e_new, vnewc,
                                                 bvc, p_new,
                                                 q_new,
                                                 ql_old, qq_old,
                                                 rho0, q_cut,
                                                 iend );

      }
      stopTimer();

      ENERGY_DATA_TEARDOWN_CUDA;

    }

    case RAJA_CUDA : {

      ENERGY_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY1;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY2;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY3;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY4;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY5;
         });

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           ibegin, iend,
           [=] __device__ (Index_type i) {
           ENERGY_BODY6;
         });

      }
      stopTimer();

      ENERGY_DATA_TEARDOWN_CUDA;

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

void ENERGY::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_e_new, getRunSize());
  checksum[vid] += calcChecksum(m_q_new, getRunSize());
}

void ENERGY::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_e_new);
  deallocData(m_e_old);
  deallocData(m_delvc);
  deallocData(m_p_new);
  deallocData(m_p_old);
  deallocData(m_q_new);
  deallocData(m_q_old);
  deallocData(m_work);
  deallocData(m_compHalfStep);
  deallocData(m_pHalfStep);
  deallocData(m_bvc);
  deallocData(m_pbvc);
  deallocData(m_ql_old);
  deallocData(m_qq_old);
  deallocData(m_vnewc);
}

} // end namespace apps
} // end namespace rajaperf
