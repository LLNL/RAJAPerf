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

///
/// ADD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   c[i] = a[i] + b[i];
/// }
///

#include "ADD.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{

#define ADD_DATA \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c;

#define ADD_BODY  \
  c[i] = a[i] + b[i];


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define ADD_DATA_SETUP_CUDA \
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend);

#define ADD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_c, c, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c)

__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                     Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     ADD_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


ADD::ADD(const RunParams& params)
  : KernelBase(rajaperf::Stream_ADD, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1000);
}

ADD::~ADD() 
{
}

void ADD::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
}

void ADD::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  switch ( vid ) {

    case Base_Seq : {

      ADD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      ADD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      ADD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      ADD_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_TARGET_OPENMP)

#define NUMTEAMS 128

    case Base_OpenMPTarget : {

      ADD_DATA;

      int n = getRunSize();
      #pragma omp target enter data map(to:a[0:n],b[0:n],c[0:n])

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        for (Index_type i = ibegin; i < iend; ++i ) {
          ADD_BODY;
        }

      }
      stopTimer();

      #pragma omp target exit data map(from:c[0:n]) map(delete:a[0:n],b[0:n])

      break;
    }

    case RAJA_OpenMPTarget : {

      ADD_DATA;

      int n = getRunSize();
      #pragma omp target enter data map(to:a[0:n],b[0:n],c[0:n])

      startTimer();
      #pragma omp target data use_device_ptr(a,b,c)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      #pragma omp target exit data map(from:c[0:n]) map(delete:a[0:n],b[0:n])

      break;
    }
#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      ADD_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         add<<<grid_size, block_size>>>( c, a, b,
                                         iend ); 

      }
      stopTimer();

      ADD_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      ADD_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ADD_BODY;
        });

      }
      stopTimer();

      ADD_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

    default : {
      std::cout << "\n  ADD : Unknown variant id = " << vid << std::endl;
    }

  }

}

void ADD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getRunSize());
}

void ADD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
