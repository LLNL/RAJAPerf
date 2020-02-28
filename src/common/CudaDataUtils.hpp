//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
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


#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"


namespace rajaperf
{

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
 * \brief Allocate CUDA device data array (dptr) and copy given hptr (host) 
 * data to device array.
 */
template <typename T>
void allocAndInitCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );

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
  dptr = 0;
}


}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard

