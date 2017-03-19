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
 * \brief Allocate and initialize 1D data array.
 */
void allocAndInitAligned(RAJA::Real_ptr ptr, int len, VariantID vid);


/*!
 * \brief Initialize 1D data array.
 */
void initData(RAJA::Real_ptr ptr, int len, VariantID vid);

/*!
 * \brief Initialize scalar data.
 */
void initData(RAJA::Real_type& d);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
