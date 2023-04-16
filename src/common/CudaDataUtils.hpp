//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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
#include <stdexcept>

#if defined(RAJA_ENABLE_CUDA)

#include "common/RAJAPerfSuite.hpp"
#include "common/GPUUtils.hpp"

#include "RAJA/policy/cuda/policy.hpp"
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


namespace detail
{

/*
 * Copy memory len bytes from src to dst.
 */
inline void copyCudaData(void* dst_ptr, const void* src_ptr, size_t len)
{
  cudaErrchk( cudaMemcpy( dst_ptr, src_ptr, len,
              cudaMemcpyDefault ) );
}

/*!
 * \brief Allocate CUDA device data array (dptr).
 */
inline void* allocCudaDeviceData(size_t len)
{
  void* dptr = nullptr;
  cudaErrchk( cudaMalloc( &dptr, len ) );
  return dptr;
}

/*!
 * \brief Allocate CUDA managed data array (dptr).
 */
inline void* allocCudaManagedData(size_t len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA pinned data array (pptr).
 */
inline void* allocCudaPinnedData(size_t len)
{
  void* pptr = nullptr;
  cudaErrchk( cudaHostAlloc( &pptr, len, cudaHostAllocMapped ) );
  return pptr;
}


/*!
 * \brief Free device data array.
 */
inline void deallocCudaDeviceData(void* dptr)
{
  cudaErrchk( cudaFree( dptr ) );
}

/*!
 * \brief Free managed data array.
 */
inline void deallocCudaManagedData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free pinned data array.
 */
inline void deallocCudaPinnedData(void* pptr)
{
  cudaErrchk( cudaFreeHost( pptr ) );
}

}  // closing brace for detail namespace


/*!
 * \brief Copy given hptr (host) data to CUDA device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initCudaDeviceData(T* dptr, const T* hptr, int len)
{
  cudaErrchk( cudaMemcpy( dptr, hptr, len * sizeof(T), cudaMemcpyHostToDevice ) );
}

/*!
 * \brief Copy given dptr (CUDA device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getCudaDeviceData(T* hptr, const T* dptr, int len)
{
  cudaErrchk( cudaMemcpy( hptr, dptr, len * sizeof(T), cudaMemcpyDeviceToHost ) );
}

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
