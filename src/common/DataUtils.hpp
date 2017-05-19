/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for utility routines for data management.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef DataUtils_HPP
#define DataUtils_HPP

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

namespace rajaperf
{

/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();

/*!
 * Increment counter for data initialization.
 */
void incDataInitCount();


/*!
 * \brief Allocate and initialize Int_type data array.
 */
void allocAndInitData(Int_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array.
 */
void allocAndInitData(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array with random sign.
 */
void allocAndInitDataRandSign(Real_ptr& ptr, int len,
                              VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Complex_type data array.
 */
void allocAndInitData(Complex_ptr& ptr, int len,
                      VariantID vid = NumVariants);


/*!
 * \brief Free data arrays.
 */
void deallocData(Int_ptr& ptr);
///
void deallocData(Real_ptr& ptr);
///
void deallocData(Complex_ptr& ptr);


/*!
 * \brief Initialize Int_type data array.
 */
void initData(Int_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array.
 */
void initData(Real_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type data array with random sign.
 */
void initDataRandSign(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);

/*!
 * \brief Initialize Complex_type data array.
 */
void initData(Complex_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize Real_type scalar data.
 */
void initData(Real_type& d,
              VariantID vid = NumVariants);


#if defined(RAJA_ENABLE_CUDA)

#if 1
/*!
 * \brief Allocate CUDA device Int_type array and copy from host to device.
 */
void allocAndInitCudaDeviceData(Int_ptr& dptr, const Int_ptr hptr, int len);

/*!
 * \brief Allocate CUDA device Real_type array and copy from host to device.
 */
void allocAndInitCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len);

#if 0 // Index_type
/*!
 * \brief Allocate CUDA device Index_type array and copy from host to device.
 */
void allocAndInitCudaDeviceData(Index_ptr& dptr, const Index_ptr hptr,
                                int len);
#endif

#else

template <typename T>
void initCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMemcpy( dptr, hptr, 
                          len * sizeof(std::remove_pointer<decltype(dptr)>::type),
                          cudaMemcpyHostToDevice ) );

  incDataInitCount();
}

template <typename T>
void allocAndInitCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMalloc( (void**)&dptr,
              len * sizeof(std::remove_pointer<decltype(dptr)>::type) ) );

  initCudaDeviceData(dptr, hptr, len);
}

template <typename T>
void getCudaDeviceData(T& hptr, const T dptr, int len)
{
  cudaErrchk( cudaMemcpy( hptr, dptr, 
              len * sizeof(std::remove_pointer<decltype(hptr)>::type),
              cudaMemcpyDeviceToHost ) );
}

template <typename T>
void deallocCudaDeviceData(T& dptr)
{
  cudaErrchk( cudaFree( dptr ) );
  dptr = 0;
}
#endif


/*!
 * \brief Copy host data Int_type array to CUDA device.
 */
void initCudaDeviceData(Int_ptr& dptr, const Int_ptr hptr, int len);

/*!
 * \brief Copy host data Real_type array to CUDA device.
 */
void initCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len);

#if 0 // Index_type
/*!
 * \brief Copy host data Index_type array to CUDA device.
 */
void initCudaDeviceData(Index_ptr& dptr, const Index_ptr hptr, int len);
#endif


/*!
 * \brief Copy CUDA device Int_type data array back to host.
 */
void getCudaDeviceData(Int_ptr& hptr, const Int_ptr dptr, int len);

/*!
 * \brief Copy CUDA device Real_type data array back to host.
 */
void getCudaDeviceData(Real_ptr& hptr, const Real_ptr dptr, int len);


/*!
 * \brief Deallocate CUDA device Int_type data array.
 */
void deallocCudaDeviceData(Int_ptr& dptr);

/*!
 * \brief Deallocate CUDA device Real_type data array.
 */
void deallocCudaDeviceData(Real_ptr& dptr);

#if 0 // Index_type 
/*!
 * \brief Deallocate CUDA device Index_type data array.
 */
void deallocCudaDeviceData(Index_ptr& dptr);
#endif

#endif


/*!
 * \brief Calculate and return checksum for data arrays.
 */
long double calcChecksum(Real_ptr d, int len, 
                         Real_type scale_factor = 1.0);
///
long double calcChecksum(Complex_ptr d, int len, 
                         Real_type scale_factor = 1.0);


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
