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

#include "RAJA/RAJA.hxx"


namespace rajaperf
{

/*!
 * Reset counter for data initialization.
 */
void resetDataInitCount();


/*!
 * \brief Allocate and initialize data array.
 */
void allocAndInitAligned(RAJA::Real_ptr ptr, int len,
                         VariantID vid = NumVariants);

/*!
 * \brief Initialize data array.
 */
void initData(RAJA::Real_ptr ptr, int len,
              VariantID vid = NumVariants);


/*!
 * \brief Initialize scalar data.
 */
void initData(RAJA::Real_type& d,
              VariantID vid = NumVariants);


/*!
 * \brief Calculate and return checksum for data array.
 */
long double calcChecksum(RAJA::Real_ptr d, int len, 
                         RAJA::Real_type scale_factor = 1.0);


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
