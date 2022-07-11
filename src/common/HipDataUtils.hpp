//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

/*!
 * \brief Get current hip device.
 */
inline int getHipDevice()
{
  int device = hipInvalidDeviceId;
  hipErrchk( hipGetDevice( &device ) );
  return device;
}

/*!
 * \brief Copy given hptr (host) data to HIP device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initHipDeviceData(T& dptr, const T hptr, int len)
{
  hipErrchk( hipMemcpy( dptr, hptr,
                          len * sizeof(typename std::remove_pointer<T>::type),
                          hipMemcpyHostToDevice ) );

  incDataInitCount();
}

/*!
 * \brief Allocate HIP device data array (dptr).
 */
template <typename T>
void allocHipDeviceData(T& dptr, int len)
{
  hipErrchk( hipMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );
}

/*!
 * \brief Allocate HIP fine-grained device data array (dfptr).
 */
template <typename T>
void allocHipDeviceFineData(T& dfptr, int len)
{
  hipErrchk( hipExtMallocWithFlags( (void**)&dfptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipDeviceMallocFinegrained ) );
}

/*!
 * \brief Allocate HIP managed data array (mptr).
 */
template <typename T>
void allocHipManagedData(T& mptr, int len)
{
  hipErrchk( hipMallocManaged( (void**)&mptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipMemAttachGlobal ) );
}

/*!
 * \brief Allocate HIP pinned data array (pptr).
 */
template <typename T>
void allocHipPinnedData(T& pptr, int len)
{
  hipErrchk( hipHostMalloc( (void**)&pptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipHostMallocMapped ) );
}

/*!
 * \brief Allocate HIP fine-grained pinned data array (pfptr).
 */
template <typename T>
void allocHipPinnedFineData(T& pfptr, int len)
{
  hipErrchk( hipHostMalloc( (void**)&pfptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipHostMallocMapped | hipHostMallocCoherent ) );
}

/*!
 * \brief Allocate HIP coarse-grained pinned data array (pcptr).
 */
template <typename T>
void allocHipPinnedCoarseData(T& pcptr, int len)
{
  hipErrchk( hipHostMalloc( (void**)&pcptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipHostMallocMapped | hipHostMallocNonCoherent ) );
}

/*!
 * \brief Apply mem advice to HIP data array (ptr).
 */
template <typename T>
void adviseHipData(T& ptr, int len, hipMemoryAdvise advice, int device)
{
  hipErrchk( hipMemAdvise( (void*)ptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              advice, device ) );
}

/*!
 * \brief Allocate HIP device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocAndInitHipDeviceData(T& dptr, const T hptr, int len)
{
  allocHipDeviceData(dptr, len);
  initHipDeviceData(dptr, hptr, len);
}

/*!
 * \brief Copy given dptr (HIP device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getHipDeviceData(T& hptr, const T dptr, int len)
{
  hipErrchk( hipMemcpy( hptr, dptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipMemcpyDeviceToHost ) );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocHipDeviceData(T& dptr)
{
  hipErrchk( hipFree( dptr ) );
  dptr = nullptr;
}

/*!
 * \brief Free managed data array.
 */
template <typename T>
void deallocHipManagedData(T& mptr)
{
  hipErrchk( hipFree( mptr ) );
  mptr = nullptr;
}

/*!
 * \brief Free pinned data array.
 */
template <typename T>
void deallocHipPinnedData(T& pptr)
{
  hipErrchk( hipHostFree( pptr ) );
  pptr = nullptr;
}


/*!
 * \brief Copy given hptr (host) data to HIP (cptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of proper size for copy operation to succeed.
 */
template <typename T>
void initHipData(HipDataSpace, T& cptr, const T hptr, int len)
{
  hipErrchk( hipMemcpy( cptr, hptr,
                          len * sizeof(typename std::remove_pointer<T>::type),
                          hipMemcpyDefault ) );

  incDataInitCount();
}

/*!
 * \brief Allocate HIP data array (cptr).
 */
template <typename T>
void allocHipData(HipDataSpace hipDataSpace, T& cptr, int len)
{
  switch (hipDataSpace) {

    case HipDataSpace::Host:
    {
      allocData(cptr, len);
    } break;
    case HipDataSpace::HostAdviseFine:
    {
      allocData(cptr, len);
      adviseHipData(cptr, len, hipMemAdviseUnsetCoarseGrain, getHipDevice());
    } break;
    case HipDataSpace::HostAdviseCoarse:
    {
      allocData(cptr, len);
      adviseHipData(cptr, len, hipMemAdviseSetCoarseGrain, getHipDevice());
    } break;
    case HipDataSpace::Pinned:
    {
      allocHipPinnedData(cptr, len);
    } break;
    case HipDataSpace::PinnedFine:
    {
      allocHipPinnedFineData(cptr, len);
    } break;
    case HipDataSpace::PinnedCoarse:
    {
      allocHipPinnedCoarseData(cptr, len);
    } break;
    case HipDataSpace::Managed:
    {
      allocHipManagedData(cptr, len);
    } break;
    case HipDataSpace::ManagedAdviseFine:
    {
      allocHipManagedData(cptr, len);
      adviseHipData(cptr, len, hipMemAdviseUnsetCoarseGrain, getHipDevice());
    } break;
    case HipDataSpace::ManagedAdviseCoarse:
    {
      allocHipManagedData(cptr, len);
      adviseHipData(cptr, len, hipMemAdviseSetCoarseGrain, getHipDevice());
    } break;
    case HipDataSpace::Device:
    {
      allocHipDeviceData(cptr, len);
    } break;
    case HipDataSpace::DeviceFine:
    {
      allocHipDeviceFineData(cptr, len);
    } break;
    default:
    {
      throw std::invalid_argument("allocHipData : Unknown memory type");
    } break;
  }
}

/*!
 * \brief Allocate HIP data array (cptr) and copy given hptr (host)
 * data to HIP array.
 */
template <typename T>
void allocAndInitHipData(HipDataSpace hipDataSpace, T& cptr, const T hptr, int len)
{
  allocHipData(hipDataSpace, cptr, len);
  initHipData(hipDataSpace, cptr, hptr, len);
}

/*!
 * \brief Free Hip data array.
 */
template <typename T>
void deallocHipData(HipDataSpace hipDataSpace, T& cptr)
{
  switch (hipDataSpace) {
    case HipDataSpace::Host:
    case HipDataSpace::HostAdviseFine:
    case HipDataSpace::HostAdviseCoarse:
    {
      deallocData(cptr);
    } break;
    case HipDataSpace::Pinned:
    case HipDataSpace::PinnedFine:
    case HipDataSpace::PinnedCoarse:
    {
      deallocHipPinnedData(cptr);
    } break;
    case HipDataSpace::Managed:
    case HipDataSpace::ManagedAdviseFine:
    case HipDataSpace::ManagedAdviseCoarse:
    {
      deallocHipManagedData(cptr);
    } break;
    case HipDataSpace::Device:
    case HipDataSpace::DeviceFine:
    {
      deallocHipDeviceData(cptr);
    } break;
    default:
    {
      throw std::invalid_argument("deallocHipData : Unknown memory type");
    } break;
  }
}

/*!
 * \brief Copy given cptr (HIP) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getHipData(HipDataSpace, T& hptr, const T cptr, int len)
{
  hipErrchk( hipMemcpy( hptr, cptr,
              len * sizeof(typename std::remove_pointer<T>::type),
              hipMemcpyDefault ) );
}

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
