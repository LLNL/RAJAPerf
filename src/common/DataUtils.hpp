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
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_DataUtils_HPP
#define RAJAPerf_DataUtils_HPP

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
void allocAndInitData(Real_ptr& ptr, int len, bool init = true,
                      VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Real_type data array with random sign.
 */
void allocAndInitDataRandSign(Real_ptr& ptr, int len, bool init = true,
                              VariantID vid = NumVariants);

/*!
 * \brief Allocate and initialize aligned Complex_type data array.
 */
void allocAndInitData(Complex_ptr& ptr, int len, bool init = true,
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


#if defined(ENABLE_CUDA)

template <typename T>
void initCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMemcpy( dptr, hptr, 
                          len * sizeof(typename std::remove_pointer<T>::type),
                          cudaMemcpyHostToDevice ) );

  incDataInitCount();
}

template <typename T>
void allocAndInitCudaDeviceData(T& dptr, const T hptr, int len)
{
  cudaErrchk( cudaMalloc( (void**)&dptr,
              len * sizeof(typename std::remove_pointer<T>::type) ) );

  initCudaDeviceData(dptr, hptr, len);
}

template <typename T>
void getCudaDeviceData(T& hptr, const T dptr, int len)
{
  cudaErrchk( cudaMemcpy( hptr, dptr, 
              len * sizeof(typename std::remove_pointer<T>::type),
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
 * \brief Calculate and return checksum for data arrays.
 */
long double calcChecksum(Real_ptr d, int len, 
                         Real_type scale_factor = 1.0);
///
long double calcChecksum(Complex_ptr d, int len, 
                         Real_type scale_factor = 1.0);


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
