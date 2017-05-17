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


namespace rajaperf
{

typedef enum SizeSpec {Mini,Small,Medium,Large,Extralarge,Specundefined} SizeSpec_T;
/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();

/*!
 * Increment counter for data initialization.
 */
void incDataInitCount();


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
void deallocData(Real_ptr& ptr);
///
void deallocData(Complex_ptr& ptr);


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

/*!
 * \brief Allocate CUDA device Real_type array and copy from host to device.
 */
void allocAndInitCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len);

/*!
 * \brief Copy host data Real_type array to CUDA device.
 */
void initCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len);

/*!
 * \brief Allocate CUDA device Index_type array and copy from host to device.
 */
void allocAndInitCudaDeviceData(Index_type*& dptr, const Index_type* hptr,
                                int len);

/*!
 * \brief Copy host data Index_type array to CUDA device.
 */
void initCudaDeviceData(Index_type*& dptr, const Index_type* hptr, int len);

/*!
 * \brief Copy CUDA device data array back to host.
 */
void getCudaDeviceData(Real_ptr& hptr, const Real_ptr dptr, int len);

/*!
 * \brief Deallocate CUDA device data.
 */
void deallocCudaDeviceData(Real_ptr& dptr);

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
