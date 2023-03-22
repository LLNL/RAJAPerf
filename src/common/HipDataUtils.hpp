//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for HIP kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_HipDataUtils_HPP
#define RAJAPerf_HipDataUtils_HPP

#include "RPTypes.hpp"
#include <stdexcept>

#if defined(RAJA_ENABLE_HIP)

#include "common/RAJAPerfSuite.hpp"
#include "common/GPUUtils.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"


namespace rajaperf
{

/*!
 * \brief Simple forall hip kernel that runs a lambda.
 */
template < typename Lambda >
__global__ void lambda_hip_forall(Index_type ibegin, Index_type iend, Lambda body)
{
  Index_type i = ibegin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < iend) {
    body(i);
  }
}
///
template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void lambda_hip_forall(Index_type ibegin, Index_type iend, Lambda body)
{
  Index_type i = ibegin + blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    body(i);
  }
}

/*!
 * \brief Simple hip kernel that runs a lambda.
 */
template < typename Lambda >
__global__ void lambda_hip(Lambda body)
{
  body();
}
///
template < size_t block_size, typename Lambda >
__launch_bounds__(block_size)
__global__ void lambda_hip(Lambda body)
{
  body();
}

/*!
 * \brief Getters for hip kernel indices.
 */
template < typename Index >
__device__ inline Index_type lambda_hip_get_index();

template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_x_direct>() {
  return threadIdx.x;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_y_direct>() {
  return threadIdx.y;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_thread_z_direct>() {
  return threadIdx.z;
}

template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_x_direct>() {
  return blockIdx.x;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_y_direct>() {
  return blockIdx.y;
}
template < >
__device__ inline Index_type lambda_hip_get_index<RAJA::hip_block_z_direct>() {
  return blockIdx.z;
}


namespace detail
{

/*!
 * \brief Get current hip device.
 */
inline int getHipDevice()
{
  int device = hipInvalidDeviceId;
  hipErrchk( hipGetDevice( &device ) );
  return device;
}

/*
 * Copy memory len bytes from src to dst.
 */
inline void copyHipData(void* dst_ptr, const void* src_ptr, size_t len)
{
  hipErrchk( hipMemcpy( dst_ptr, src_ptr, len,
             hipMemcpyDefault ) );
}

/*!
 * \brief Allocate HIP device data array (dptr).
 */
inline void* allocHipDeviceData(size_t len)
{
  void* dptr = nullptr;
  hipErrchk( hipMalloc( &dptr, len ) );
  return dptr;
}

/*!
 * \brief Allocate HIP fine-grained device data array (dfptr).
 */
inline void* allocHipDeviceFineData(size_t len)
{
  void* dfptr = nullptr;
  hipErrchk( hipExtMallocWithFlags( &dfptr, len,
              hipDeviceMallocFinegrained ) );
  return dfptr;
}

/*!
 * \brief Allocate HIP managed data array (mptr).
 */
inline void* allocHipManagedData(size_t len)
{
  void* mptr = nullptr;
  hipErrchk( hipMallocManaged( &mptr, len,
              hipMemAttachGlobal ) );
  return mptr;
}

/*!
 * \brief Allocate HIP pinned data array (pptr).
 */
inline void* allocHipPinnedData(size_t len)
{
  void* pptr = nullptr;
  hipErrchk( hipHostMalloc( &pptr, len,
              hipHostMallocMapped ) );
  return pptr;
}

/*!
 * \brief Allocate HIP fine-grained pinned data array (pfptr).
 */
inline void* allocHipPinnedFineData(size_t len)
{
  void* pfptr = nullptr;
  hipErrchk( hipHostMalloc( &pfptr, len,
              hipHostMallocMapped | hipHostMallocCoherent ) );
  return pfptr;
}

/*!
 * \brief Allocate HIP coarse-grained pinned data array (pcptr).
 */
inline void* allocHipPinnedCoarseData(size_t len)
{
  void* pcptr = nullptr;
  hipErrchk( hipHostMalloc( &pcptr, len,
              hipHostMallocMapped | hipHostMallocNonCoherent ) );
  return pcptr;
}

/*!
 * \brief Apply mem advice to HIP data array (ptr).
 */
inline void adviseHipData(void* ptr, int len, hipMemoryAdvise advice, int device)
{
  hipErrchk( hipMemAdvise( ptr, len, advice, device ) );
}

inline void adviseHipCoarseData(void* ptr, size_t len)
{
  adviseHipData(ptr, len, hipMemAdviseSetCoarseGrain, getHipDevice());
}

inline void adviseHipFineData(void* ptr, size_t len)
{
  adviseHipData(ptr, len, hipMemAdviseUnsetCoarseGrain, getHipDevice());
}


/*!
 * \brief Free device data array.
 */
inline void deallocHipDeviceData(void* dptr)
{
  hipErrchk( hipFree( dptr ) );
}

/*!
 * \brief Free managed data array.
 */
inline void deallocHipManagedData(void* mptr)
{
  hipErrchk( hipFree( mptr ) );
}

/*!
 * \brief Free pinned data array.
 */
inline void deallocHipPinnedData(void* pptr)
{
  hipErrchk( hipHostFree( pptr ) );
}

}  // closing brace for detail namespace


/*!
 * \brief Copy given hptr (host) data to HIP device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initHipDeviceData(T* dptr, const T* hptr, int len)
{
  hipErrchk( hipMemcpy( dptr, hptr, len * sizeof(T), hipMemcpyHostToDevice ) );
}
/*!
 * \brief Copy given dptr (HIP device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getHipDeviceData(T* hptr, const T* dptr, int len)
{
  hipErrchk( hipMemcpy( hptr, dptr, len * sizeof(T), hipMemcpyDeviceToHost ) );
}

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
