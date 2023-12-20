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
 * \brief Method for launching a CUDA kernel with given configuration.
 *
 *        Note: method checks whether number of args and their types in
 *              kernel signature matches args passed to this method.
 */
template <typename... Args, typename...KernArgs>
void RPlaunchCudaKernel(void (*kernel)(KernArgs...),
                        const dim3& numBlocks, const dim3& dimBlocks,
                        std::uint32_t sharedMemBytes, cudaStream_t stream,
                        Args const&... args)
{
  static_assert(sizeof...(KernArgs) == sizeof...(Args),
                "Number of kernel args doesn't match what's passed to method");

  static_assert(conjunction<std::is_same<std::decay_t<KernArgs>, std::decay_t<Args>>...>::value,
                "Kernel arg types don't match what's passed to method");

  constexpr size_t count = sizeof...(Args);
  void* arg_arr[count]{(void*)&args...};

  auto k = reinterpret_cast<const void*>(kernel);
  cudaLaunchKernel(k, numBlocks, dimBlocks,
                   arg_arr,
                   sharedMemBytes, stream);
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


namespace detail
{

/*!
 * \brief Get current cuda device.
 */
inline int getCudaDevice()
{
  int device = -1;
  cudaErrchk( cudaGetDevice( &device ) );
  return device;
}

/*!
 * \brief Get properties of the current cuda device.
 */
inline cudaDeviceProp getCudaDeviceProp()
{
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, getCudaDevice()));
  return prop;
}

/*!
 * \brief Get max occupancy in blocks for the given kernel for the current
 *        cuda device.
 */
template < typename Func >
RAJA_INLINE
int getCudaOccupancyMaxBlocks(Func&& func, int num_threads, size_t shmem_size)
{
  int max_blocks = -1;
  cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, func, num_threads, shmem_size));

  size_t multiProcessorCount = getCudaDeviceProp().multiProcessorCount;

  return max_blocks * multiProcessorCount;
}

/*
 * Copy memory len bytes from src to dst.
 */
inline void copyCudaData(void* dst_ptr, const void* src_ptr, Size_type len)
{
  cudaErrchk( cudaMemcpy( dst_ptr, src_ptr, len,
              cudaMemcpyDefault ) );
}

/*!
 * \brief Allocate CUDA device data array.
 */
inline void* allocCudaDeviceData(Size_type len)
{
  void* dptr = nullptr;
  cudaErrchk( cudaMalloc( &dptr, len ) );
  return dptr;
}

/*!
 * \brief Allocate CUDA managed data array.
 */
inline void* allocCudaManagedData(Size_type len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA managed host preferred data array.
 */
inline void* allocCudaManagedHostPreferredData(Size_type len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA managed device preferred data array.
 */
inline void* allocCudaManagedDevicePreferredData(Size_type len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetPreferredLocation, getCudaDevice() ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA managed host preferred host accessed data array.
 */
inline void* allocCudaManagedHostPreferredDeviceAccessedData(Size_type len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetAccessedBy, getCudaDevice() ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA managed device preferred host accessed data array.
 */
inline void* allocCudaManagedDevicePreferredHostAccessedData(Size_type len)
{
  void* mptr = nullptr;
  cudaErrchk( cudaMallocManaged( &mptr, len, cudaMemAttachGlobal ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetPreferredLocation, getCudaDevice() ) );
  cudaErrchk( cudaMemAdvise( mptr, len, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId ) );
  return mptr;
}

/*!
 * \brief Allocate CUDA pinned data array.
 */
inline void* allocCudaPinnedData(Size_type len)
{
  void* pptr = nullptr;
  cudaErrchk( cudaHostAlloc( &pptr, len, cudaHostAllocMapped ) );
  return pptr;
}


/*!
 * \brief Free CUDA device data array.
 */
inline void deallocCudaDeviceData(void* dptr)
{
  cudaErrchk( cudaFree( dptr ) );
}

/*!
 * \brief Free CUDA managed data array.
 */
inline void deallocCudaManagedData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free CUDA managed host preferred data array.
 */
inline void deallocCudaManagedHostPreferredData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free CUDA managed device preferred data array.
 */
inline void deallocCudaManagedDevicePreferredData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free CUDA managed host preferred host accessed data array.
 */
inline void deallocCudaManagedHostPreferredDeviceAccessedData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free CUDA managed device preferred host accessed data array.
 */
inline void deallocCudaManagedDevicePreferredHostAccessedData(void* mptr)
{
  cudaErrchk( cudaFree( mptr ) );
}

/*!
 * \brief Free CUDA pinned data array.
 */
inline void deallocCudaPinnedData(void* pptr)
{
  cudaErrchk( cudaFreeHost( pptr ) );
}

}  // closing brace for detail namespace

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
