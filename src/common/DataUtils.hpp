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
 * \brief Allocate and initialize aligned data arrays.
 */
void allocAndInitData(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);
///
void allocAndInitDataRandSign(Real_ptr& ptr, int len,
                              VariantID vid = NumVariants);
///
void allocAndInitData(Complex_ptr& ptr, int len,
                      VariantID vid = NumVariants);
///
#if defined(RAJA_ENABLE_CUDA)
void allocAndInitCudaDeviceData(Real_ptr& dptr, const Real_ptr hptr, int len);
#endif


/*!
 * \brief Copy CUDA device data array back to host.
 */
#if defined(RAJA_ENABLE_CUDA)
void getCudaDeviceData(Real_ptr& hptr, const Real_ptr dptr, int len);
#endif


/*!
 * \brief Free data arrays.
 */
void deallocData(Real_ptr& ptr);
///
void deallocData(Complex_ptr& ptr);
///
#if defined(RAJA_ENABLE_CUDA)
void deallocCudaDeviceData(Real_ptr& dptr);
#endif


/*!
 * \brief Initialize aligned data arrays.
 */
void initData(Real_ptr& ptr, int len,
              VariantID vid = NumVariants);
///
void initDataRandSign(Real_ptr& ptr, int len,
                      VariantID vid = NumVariants);
///
void initData(Complex_ptr& ptr, int len,
              VariantID vid = NumVariants);

/*!
 * \brief Initialize scalar data.
 */
void initData(Real_type& d,
              VariantID vid = NumVariants);


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
