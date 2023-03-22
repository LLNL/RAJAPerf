//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods for openmp target kernel data allocation, initialization,
/// and deallocation.
///


#ifndef RAJAPerf_OpenMPTargetDataUtils_HPP
#define RAJAPerf_OpenMPTargetDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)


namespace rajaperf
{

namespace detail
{


/*
 * Copy memory len bytes from src to dst.
 */
inline void copyOpenMPTargetData(void* dst_ptr, const void* src_ptr, size_t len,
                          int dst_did, int src_did)
{
  omp_target_memcpy( dst_ptr, src_ptr, len,
                     0, 0, dst_did, src_did );
}

/*!
 * \brief Copy given hptr (host) data to device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void copyOpenMPDeviceData(T& dptr, const T hptr, int len,
                          int did = omp_get_default_device(),
                          int hid = omp_get_initial_device())
{
  omp_target_memcpy( dptr, hptr,
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, did, hid );
}

/*!
 * \brief Copy given hptr (host) data to device (dptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void initOpenMPDeviceData(T& dptr, const T hptr, int len,
                          int did = omp_get_default_device(),
                          int hid = omp_get_initial_device())
{
  copyOpenMPDeviceData(dptr, hptr, len, did, hid);
}

/*!
 * \brief Allocate device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocOpenMPDeviceData(T& dptr, int len,
                           int did = omp_get_default_device())
{
  dptr = static_cast<T>( omp_target_alloc(
                         len * sizeof(typename std::remove_pointer<T>::type),
                         did) );
}

/*!
 * \brief Allocate device data array (dptr) and copy given hptr (host)
 * data to device array.
 */
template <typename T>
void allocAndInitOpenMPDeviceData(T& dptr, const T hptr, int len,
                                  int did = omp_get_default_device(),
                                  int hid = omp_get_initial_device())
{
  allocOpenMPDeviceData(dptr, len, did);
  initOpenMPDeviceData(dptr, hptr, len, did, hid);
}

/*!
 * \brief Copy given device ptr (dptr) data to host ptr (hptr).
 *
 * Method assumes both host and device data arrays are allocated
 * and of propoer size for copy operation to succeed.
 */
template <typename T>
void getOpenMPDeviceData(T& hptr, const T dptr, int len,
                         int hid = omp_get_initial_device(),
                         int did = omp_get_default_device())
{
  omp_target_memcpy( hptr, dptr,
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, hid, did );
}

/*!
 * \brief Free device data array.
 */
template <typename T>
void deallocOpenMPDeviceData(T& dptr,
                             int did = omp_get_default_device())
{
  omp_target_free( dptr, did );
  dptr = 0;
}


}  // closing brace for detail namespace

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_TARGET_OPENMP

#endif  // closing endif for header file include guard
