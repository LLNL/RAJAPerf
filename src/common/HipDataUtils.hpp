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

#if defined(RAJA_ENABLE_HIP)

#include "common/GPUUtils.hpp"

#include "RAJA/policy/hip/raja_hiperrchk.hpp"


namespace rajaperf
{

/*!
 * \brief Get hip arch name.
 */
inline std::string getHipArchName()
{
  int dev = -1;
  hipDeviceProp_t devProp;
  hipErrchk(hipGetDevice(&dev));
  hipErrchk(hipGetDeviceProperties(&devProp, dev));
  return devProp.gcnArchName;
}

#if defined(__gfx90a__)
// NOTE: this will only be defined while compiling device code
#define RAJAPERF_HIP_unsafeAtomicAdd \
  ::unsafeAtomicAdd
#else
#define RAJAPERF_HIP_unsafeAtomicAdd \
  ignore_unused
#endif

/*!
 * \brief Check if compiled code with unsafe atomics.
 */
inline bool haveHipUnsafeAtomics()
{
  std::string hipArch = getHipArchName();
#if defined(RP_USE_DOUBLE)
  if (hipArch.find("gfx90a") == 0) {
    return true;
  }
#endif
#if defined(RP_USE_FLOAT)
  if (hipArch.find("gfx90a") == 0) {
    return true;
  }
#endif
  return false;
}


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
 * \brief Free pinned data array.
 */
template <typename T>
void deallocHipPinnedData(T& pptr)
{
  hipErrchk( hipHostFree( pptr ) );
  pptr = nullptr;
}


}  // closing brace for rajaperf namespace


#define RAJAPERF_REDUCE_1_HIP(type, make_val, dst, init, op, atomicOp) \
  \
  HIP_DYNAMIC_SHARED(type, _shmem); \
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
  for ( unsigned i = block_size / 2u; i > 0u; i /= 2u ) { \
    if ( threadIdx.x < i ) { \
      _shmem[ threadIdx.x ] = op(_shmem[ threadIdx.x ], _shmem[ threadIdx.x + i ]); \
    } \
     __syncthreads(); \
  } \
  \
  if ( threadIdx.x == 0 ) { \
    atomicOp( dst, _shmem[ 0 ] ); \
  }

#define RAJAPERF_REDUCE_3_HIP(type, make_vals, dst0, init0, op0, atomicOp0, \
                                               dst1, init1, op1, atomicOp1, \
                                               dst2, init2, op2, atomicOp2) \
  \
  HIP_DYNAMIC_SHARED(type, _shmem); \
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
  for ( unsigned i = block_size / 2u; i > 0u; i /= 2u ) { \
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

#endif // RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
