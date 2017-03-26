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

#ifndef DataUtils_HXX
#define DataUtils_HXX

#include "RAJAPerfSuite.hxx"
#include "RPTypes.hxx"


namespace rajaperf
{

/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();


/*!
 * \brief Allocate and initialize aligned data arrays.
 */
void allocAndInit(Real_ptr& ptr, int len,
                  VariantID vid = NumVariants);
///
void allocAndInit(Complex_ptr& ptr, int len,
                  VariantID vid = NumVariants);

/*!
 * \brief Free data arrays.
 */
void dealloc(Real_ptr& ptr);
///
void dealloc(Complex_ptr& ptr);


/*!
 * \brief Initialize aligned data arrays.
 */
void initData(Real_ptr& ptr, int len,
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
