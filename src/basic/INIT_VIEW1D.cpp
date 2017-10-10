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
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = 1.5;
/// }
///
/// RAJA variants use a "view" and an "offset layout" to do the same thing
/// where the loop runs over an offset range.
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
  Real_ptr a = m_a;

#define INIT_VIEW1D_DATA_RAJA \
  Real_ptr a = m_a; \
\
  ViewType view(a, RAJA::make_offset_layout<1>({{1}}, {{iend+1}}));


#define INIT_VIEW1D_BODY  \
  a[i] = 0.123;

#define INIT_VIEW1D_BODY_RAJA  \
  view(i) = 0.123;


#if defined(RAJA_ENABLE_CUDA)

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_DATA_SETUP_CUDA \
  Real_ptr a; \
\
  allocAndInitCudaDeviceData(a, m_a, iend);

#define INIT_VIEW1D_DATA_SETUP_CUDA_RAJA \
  Real_ptr a; \
\
  allocAndInitCudaDeviceData(a, m_a, iend); \
\
  ViewType view(a, RAJA::make_offset_layout<1>({{1}}, {{iend+1}}));


#define INIT_VIEW1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, iend); \
  deallocCudaDeviceData(a);

__global__ void initview1d(Real_ptr a, 
                           Index_type iend) 
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
  allocAndInitData(m_a, getRunSize(), vid);
}

void INIT_VIEW1D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  using ViewType = RAJA::View<Real_type, RAJA::OffsetLayout<1> >;

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
          RAJA::RangeSegment(ibegin+1, iend+1), 
          [=](Index_type i) {
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

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

      INIT_VIEW1D_DATA_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin+1, iend+1), 
          [=](Index_type i) {
          INIT_VIEW1D_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      INIT_VIEW1D_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

         const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
         initview1d<<<grid_size, block_size>>>( a,
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
           RAJA::RangeSegment(ibegin+1, iend+1),
           [=] __device__ (Index_type i) {
           INIT_VIEW1D_BODY_RAJA;
         });

      }
      stopTimer();

      INIT_VIEW1D_DATA_TEARDOWN_CUDA;

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
