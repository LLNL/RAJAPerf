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
/// INIT_VIEW1D kernel reference implementation:
///
/// const Real_type val = ...;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = val;
/// }
///
/// RAJA variants use a "view" and "layout" to do the same thing
/// where the loop runs over the same range.
///

#include "INIT_VIEW1D.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define INIT_VIEW1D_DATA \
  Real_ptr a = m_a; \
  const Real_type v = m_val;

#define INIT_VIEW1D_DATA_RAJA \
  Real_ptr a = m_a; \
  const Real_type v = m_val; \
\
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);


#define INIT_VIEW1D_BODY  \
  a[i] = v;

#define INIT_VIEW1D_BODY_RAJA  \
  view(i) = v;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_DATA_SETUP_CUDA \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitCudaDeviceData(a, m_a, iend);

#define INIT_VIEW1D_DATA_SETUP_CUDA_RAJA \
  Real_ptr a; \
  const Real_type v = m_val; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
\
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);


#define INIT_VIEW1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, iend); \
  deallocCudaDeviceData(a);

__global__ void initview1d(Real_ptr a, 
                           Real_type v,
                           const Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     INIT_VIEW1D_BODY; 
   }
}

#endif // if defined(RAJA_ENABLE_CUDA)


INIT_VIEW1D::INIT_VIEW1D(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT_VIEW1D, params)
{
   setDefaultSize(500000);
   setDefaultReps(5000);
}

INIT_VIEW1D::~INIT_VIEW1D() 
{
}

void INIT_VIEW1D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  m_val = 0.123;
}

void INIT_VIEW1D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  using ViewType = RAJA::View<Real_type, RAJA::Layout<1> >;

  switch ( vid ) {

    case Base_Seq : {

      INIT_VIEW1D_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INIT_VIEW1D_DATA_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT_VIEW1D_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      INIT_VIEW1D_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      INIT_VIEW1D_DATA_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT_VIEW1D_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }


#if defined(RAJA_ENABLE_TARGET_OPENMP)

#define NUMTEAMS 128

    case Base_OpenMPTarget : {

      INIT_VIEW1D_DATA                 

      #pragma omp target enter data map(to:a[0:iend],v)

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp target teams distribute parallel for num_teams(NUMTEAMS) schedule(static, 1) 
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      #pragma omp target exit data map(from:a[0:iend]) map(delete:v)

      break;
    }

    case RAJA_OpenMPTarget : {

#if 0 
      INIT_VIEW1D_DATA_RAJA                 
      #pragma omp target enter data map(to:a[0:iend],v)

      startTimer();
      #pragma omp target data use_device_ptr(a)
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_target_parallel_for_exec<NUMTEAMS>>(RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT_VIEW1D_BODY_RAJA;;
        });

      }
      stopTimer();

      #pragma omp target exit data map(from:a[0:iend]) map(delete:v)
#endif // still need to figure out how layout and view will work under RAJA omp-target

      break;
    }

#endif //RAJA_ENABLE_TARGET_OPENMP
#endif //RAJA_ENABLE_OMP                             

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      INIT_VIEW1D_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         initview1d<<<grid_size, block_size>>>( a,
                                                v, 
                                                iend ); 

      }
      stopTimer();

      INIT_VIEW1D_DATA_TEARDOWN_CUDA;

      break; 
    }

    case RAJA_CUDA : {

      INIT_VIEW1D_DATA_SETUP_CUDA_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
           RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
           INIT_VIEW1D_BODY_RAJA;
         });

      }
      stopTimer();

      INIT_VIEW1D_DATA_TEARDOWN_CUDA;

      break;
    }
#endif

    default : {
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INIT_VIEW1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void INIT_VIEW1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
}

} // end namespace basic
} // end namespace rajaperf
