//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for CUDA kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_CudaDataUtils_HPP
#define RAJAPerf_CudaDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/GPUUtils.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"


namespace rajaperf
{

/*!
 * \brief Device timer, returns a time in ns from an arbitrary starting point.
 * Note that this time is consistent across the whole device.
 */
__device__ __forceinline__ unsigned long long device_timer()
{
  unsigned long long global_timer = 0;
#if __CUDA_ARCH__ >= 300
  asm volatile ("mov.u64 %0, %globaltimer;" : "=l"(global_timer));
#endif
  return global_timer;
}

/*!
 * \brief Simple forall cuda kernel that runs a lambda.
 */
template < typename Lambda >
__global__ void lambda_cuda_forall(Index_type ibegin, Index_type iend, Lambda body)
{
  Index_type i = ibegin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    body(i);
  }
}
///
template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void lambda_cuda_forall(Index_type ibegin, Index_type iend, Lambda body)
{
  Index_type i = ibegin + blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    body(i);
  }
}

/*!
 * \brief Simple cuda kernel that runs a lambda.
 */
template < typename Lambda >
__global__ void lambda_cuda(Lambda body)
{
  body();
}
///
template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void lambda_cuda(Lambda body)
{
  body();
}

/*!
 * \brief Getters for cuda kernel indices.
 */
template < typename Index >
__device__ inline Index_type lambda_cuda_get_index();

template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_thread_x_direct>() {
  return threadIdx.x;
}
template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_thread_y_direct>() {
  return threadIdx.y;
}
template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_thread_z_direct>() {
  return threadIdx.z;
}

template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_block_x_direct>() {
  return blockIdx.x;
}
template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_block_y_direct>() {
  return blockIdx.y;
}
template < >
__device__ inline Index_type lambda_cuda_get_index<RAJA::cuda_block_z_direct>() {
  return blockIdx.z;
}

/*!
 * \brief Copy given hptr (host) data to CUDA device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMemcpy( dptr, hptr,
                          len * sizeof(typename std::remove_pointer<T>::type),
                          cudaMemcpyHostToDevice ) );

  incDataInitCount();
}

/*!
 * \brief Allocate CUDA device data array (dptr).
 */
template <typename T>
void allocCudaDeviceData(T& dptr, int len)
{
  cudaErrchk( cudaMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );
}

/*!
 * \brief Allocate CUDA pinned data array (pptr).
 */
template <typename T>
void allocCudaPinnedData(T& pptr, int len)
{
  cudaErrchk( cudaHostAlloc( (void**)&pptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              cudaHostAllocMapped ) );
}

/*!
 * \brief Allocate CUDA device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocAndInitCudaDeviceData(T& dptr, const T hptr, int len)
{
  allocCudaDeviceData(dptr, len);
  initCudaDeviceData(dptr, hptr, len);
}

/*!
 * \brief Copy given dptr (CUDA device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getCudaDeviceData(T& hptr, const T dptr, int len)
{
  cudaErrchk( cudaMemcpy( hptr, dptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              cudaMemcpyDeviceToHost ) );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocCudaDeviceData(T& dptr)
{
  cudaErrchk( cudaFree( dptr ) );
  dptr = nullptr;
}

/*!
 * \brief Free pinned data array.
 */
template <typename T>
void deallocCudaPinnedData(T& pptr)
{
  cudaErrchk( cudaFreeHost( pptr ) );
  pptr = nullptr;
}

}  // closing brace for rajaperf namespace


#define RAJAPERF_REDUCE_1_CUDA(type, make_val, dst, init, op, atomicOp) \
  \
  extern __shared__ type _shmem[ ]; \
  \
  _shmem[ threadIdx.x ] = init; \
  \
  for ( Index_type i = blockIdx.x * block_size + threadIdx.x; \
        i < iend ; i += gridDim.x * block_size ) { \
    make_val; \
    _shmem[ threadIdx.x ] = op(_shmem[ threadIdx.x ], val); \
  } \
  __syncthreads(); \
  \
  for ( int i = block_size / 2; i > 0; i /= 2 ) { \
    if ( threadIdx.x < i ) { \
      _shmem[ threadIdx.x ] = op(_shmem[ threadIdx.x ], _shmem[ threadIdx.x + i ]); \
    } \
     __syncthreads(); \
  } \
  \
  if ( threadIdx.x == 0 ) { \
    atomicOp( dst, _shmem[ 0 ] ); \
  }

#define RAJAPERF_REDUCE_3_CUDA(type, make_vals, dst0, init0, op0, atomicOp0, \
                                                dst1, init1, op1, atomicOp1, \
                                                dst2, init2, op2, atomicOp2) \
  \
  extern __shared__ type _shmem[ ]; \
  \
  type* _shmem0 = _shmem + 0 * block_size; \
  type* _shmem1 = _shmem + 1 * block_size; \
  type* _shmem2 = _shmem + 2 * block_size; \
  \
  _shmem0[ threadIdx.x ] = init0; \
  _shmem1[ threadIdx.x ] = init1; \
  _shmem2[ threadIdx.x ] = init2; \
  \
  for ( Index_type i = blockIdx.x * block_size + threadIdx.x; \
        i < iend ; i += gridDim.x * block_size ) { \
    make_vals; \
    _shmem0[ threadIdx.x ] = op0(_shmem0[ threadIdx.x ], val0); \
    _shmem1[ threadIdx.x ] = op1(_shmem1[ threadIdx.x ], val1); \
    _shmem2[ threadIdx.x ] = op2(_shmem2[ threadIdx.x ], val2); \
  } \
  __syncthreads(); \
  \
  for ( int i = block_size / 2; i > 0; i /= 2 ) { \
    if ( threadIdx.x < i ) { \
      _shmem0[ threadIdx.x ] = op0(_shmem0[ threadIdx.x ], _shmem0[ threadIdx.x + i ]); \
      _shmem1[ threadIdx.x ] = op1(_shmem1[ threadIdx.x ], _shmem1[ threadIdx.x + i ]); \
      _shmem2[ threadIdx.x ] = op2(_shmem2[ threadIdx.x ], _shmem2[ threadIdx.x + i ]); \
    } \
     __syncthreads(); \
  } \
  \
  if ( threadIdx.x == 0 ) { \
    atomicOp0( dst0, _shmem0[ 0 ] ); \
    atomicOp1( dst1, _shmem1[ 0 ] ); \
    atomicOp2( dst2, _shmem2[ 0 ] ); \
  }

#endif // RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
