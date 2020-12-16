//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for SYCL kernel data allocation, initialization, and deallocation.
///


#ifndef RAJAPerf_SyclDataUtils_HPP
#define RAJAPerf_SyclDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_SYCL)


#include <CL/sycl.hpp>


namespace rajaperf
{

/*!
 * \brief Copy given hptr (host) data to SYCL device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initSyclDeviceData(T& dptr, const T hptr, int len, cl::sycl::queue qu)
{
  auto e = qu.memcpy( dptr, hptr,
                      len * sizeof(typename std::remove_pointer<T>::type));
  e.wait();

  incDataInitCount();
}

/*!
 * \brief Allocate SYCL device data array (dptr) and copy given hptr (host) 
 * data to device array.
 */
template <typename T>
void allocAndInitSyclDeviceData(T& dptr, const T hptr, int len, cl::sycl::queue qu)
{
  dptr = cl::sycl::malloc_device<typename std::remove_pointer<T>::type>(len, qu);

  initSyclDeviceData(dptr, hptr, len, qu);
}

/*!
 * \brief Copy given dptr (SYCL device) data to host (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getSyclDeviceData(T& hptr, const T dptr, int len, cl::sycl::queue qu)
{
  auto e = qu.memcpy( hptr, dptr,
                      len * sizeof(typename std::remove_pointer<T>::type));
  e.wait();
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocSyclDeviceData(T& dptr, cl::sycl::queue qu)
{
  cl::sycl::free(dptr, qu);
  dptr = 0;
}


}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard

