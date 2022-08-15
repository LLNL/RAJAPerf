//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_VEC_MULT_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{



#define MAT_VEC_MULT_4GROUPS_DATA_SETUP_CUDA   \
  Real_ptr d_x,d_y,d_a;                           \
  allocAndInitCudaDeviceData( d_x, m_x, iend*16); \
  allocAndInitCudaDeviceData( d_y, m_y, iend*16); \
  allocAndInitCudaDeviceData( d_a, m_a, iend*16); \
  Real_type * RAJA_RESTRICT  x = d_x;    \
  Real_type * RAJA_RESTRICT  y = d_y;    \
  Real_type * RAJA_RESTRICT  a = d_a;    

#define MAT_VEC_MULT_4GROUPS_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, d_y, iend*16);         \
  deallocCudaDeviceData(d_x);                   \
  deallocCudaDeviceData(d_y);                   \
  deallocCudaDeviceData(d_a);

#define MAT_VEC_MULT_4GROUPS_BODY_DUMMY   \


#define MAT_VEC_MULT_4GROUPS_BODY_CUDA       \
    auto aa   = Mat ( a + 16 * i );    \
    auto x0   = Vec ( x + 16 * i + 0 );   \
    auto x1   = Vec ( x + 16 * i + 4 );   \
    auto x2   = Vec ( x + 16 * i + 8 );   \
    auto x3   = Vec ( x + 16 * i +12 );   \
    auto y0   = Vec ( y + 16 * i + 0 );   \
    auto y1   = Vec ( y + 16 * i + 4 );   \
    auto y2   = Vec ( y + 16 * i + 8 );   \
    auto y3   = Vec ( y + 16 * i +12 );   \
    y0(vall) = aa(0,0) * x0(vall) + aa(0,1) * x1(vall) + aa(0,2) * x2(vall) + aa(0,3) * x3(vall); \
    y1(vall) = aa(1,0) * x0(vall) + aa(1,1) * x1(vall) + aa(1,2) * x2(vall) + aa(1,3) * x3(vall); \
    y2(vall) = aa(2,0) * x0(vall) + aa(2,1) * x1(vall) + aa(2,2) * x2(vall) + aa(2,3) * x3(vall); \
    y3(vall) = aa(3,0) * x0(vall) + aa(3,1) * x1(vall) + aa(3,2) * x2(vall) + aa(3,3) * x3(vall);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void mvm4g(Real_type* RAJA_RESTRICT  y, Real_type* RAJA_RESTRICT  x, Real_type* RAJA_RESTRICT  a,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
       auto A = a + 16*i;
       auto X = x + 16*i;
       auto Y = y + 16*i;
       for(Index_type j=0; j<4; j++){
           for(Index_type l=0; l<4; l++){
               Y[j*4+l] = 0;
           }
           for(Index_type k=0; k<4; k++){
               for(Index_type l=0; l<4; l++){
                   Y[j*4+l] += A[k*4+j] * X[4*k+l];
               }
           }
       }
   }
}


template < size_t block_size >
void MAT_VEC_MULT_4GROUPS::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MAT_VEC_MULT_4GROUPS_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MAT_VEC_MULT_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      mvm4g<block_size><<<grid_size, block_size>>>( y, x, a,
                                        iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MAT_VEC_MULT_4GROUPS_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    MAT_VEC_MULT_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        MAT_VEC_MULT_4GROUPS_BODY_CUDA;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    MAT_VEC_MULT_4GROUPS_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    MAT_VEC_MULT_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true > >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MAT_VEC_MULT_4GROUPS_BODY_CUDA;
      });

    }
    stopTimer();

    MAT_VEC_MULT_4GROUPS_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  MAT_VEC_MULT_4GROUPS : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_VEC_MULT_4GROUPS, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
