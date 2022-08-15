//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD_4GROUPS.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define ADD_4GROUPS_DATA_SETUP_CUDA              \
  Real_ptr d_a,d_b,d_c;                          \
  allocAndInitCudaDeviceData(d_a, m_a, iend*64); \
  allocAndInitCudaDeviceData(d_b, m_b, iend*16); \
  allocAndInitCudaDeviceData(d_c, m_c, iend*64); \
  Real_type * RAJA_RESTRICT  a = d_a;             \
  Real_type * RAJA_RESTRICT  b = d_b;             \
  Real_type * RAJA_RESTRICT  c = d_c;

#define ADD_4GROUPS_DATA_TEARDOWN_CUDA           \
  getCudaDeviceData(m_c, d_c, iend*64);          \
  deallocCudaDeviceData(d_a);                    \
  deallocCudaDeviceData(d_b);                    \
  deallocCudaDeviceData(d_c);


#define ADD_4GROUPS_BODY_CUDA                    \
    using mat_t = RAJA::expt::RectMatrixRegister<      \
                      Real_type,                       \
                      RAJA::expt::RowMajorLayout,      \
                      16,4                             \
                  >;                                   \
    using row_t = RAJA::expt::RowIndex<int, mat_t>;    \
    using col_t = RAJA::expt::ColIndex<int, mat_t>;    \
    using vec_t = RAJA::expt::VectorRegister<Real_type>;\
    using idx_t = RAJA::expt::VectorIndex<int, vec_t>; \
    using Mat = RAJA::View<Real_type,RAJA::StaticLayout<RAJA::PERM_IJ,16,4>>; \
    using Vec = RAJA::View<Real_type,RAJA::StaticLayout<RAJA::PERM_I ,16>>;   \
    				                       \
    auto rall = row_t::static_all();                   \
    auto call = col_t::static_all();                   \
                                                       \
    Real_type y[16*4];                                 \
    auto aa   = Mat( a + 64 * i );                     \
    auto bV   = Vec( b + 16 * i );                     \
    auto cc   = Mat( c + 64 * i );                     \
    auto yy   = Mat( y );                              \
    for(int j=0; j<16; j++){                           \
        for(int k=0; k<4; k++){                        \
            yy(j,k) = bV(j);                           \
        }                                              \
    }                                                  \
    cc(rall,call) = yy(rall,call) + aa(rall,call);


template < size_t block_size >
__launch_bounds__(block_size)
__global__ void add_4g(Real_type* RAJA_RESTRICT  a, Real_type* RAJA_RESTRICT  b, Real_type* RAJA_RESTRICT  c,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
            auto A = a + 64*i;
            auto B = b + 16*i;
            auto C = c + 64*i;
            for(Index_type j=0; j<16; j++){
                for(Index_type k=0; k<4; k++){
                    C[j*4+k] = A[j*4+k] + B[j];
                }
            }
   }
}


template < size_t block_size >
void ADD_4GROUPS::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();


  if ( vid == Base_CUDA ) {

    ADD_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      add_4g<block_size><<<grid_size, block_size>>>( a, b, c,
                                        iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    ADD_4GROUPS_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    ADD_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        ADD_4GROUPS_BODY_CUDA;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    ADD_4GROUPS_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    ADD_4GROUPS_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true > >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_4GROUPS_BODY_CUDA;
      });

    }
    stopTimer();

    ADD_4GROUPS_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  ADD_4GROUPS : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(ADD_4GROUPS, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
