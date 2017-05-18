/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file that defines data types used in performance suite.
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

#ifndef RP_Types_HPP
#define RP_Types_HPP

#include "RAJA/util/types.hpp"

//
// Only one of the following (double or float) should be defined.
// 
#define RP_USE_DOUBLE
//#undef RP_USE_DOUBLE
//#define RP_USE_FLOAT
#undef RP_USE_FLOAT

#define RP_USE_COMPLEX
//#undef RP_USE_DOUBLE

#if defined(RP_USE_COMPLEX)
#include <complex>
#endif


namespace rajaperf
{


/*!
 ******************************************************************************
 *
 * \brief Type used for indexing in all kernel sample loops.
 *
 * It is volatile to ensure that kernsls will not be optimized away by 
 * compilers, which can happen in some circumstances.
 *
 ******************************************************************************
 */
typedef volatile int SampIndex_type;


/*!
 ******************************************************************************
 *
 * \brief Types used for all kernel loop indexing.
 *
 ******************************************************************************
 */
typedef RAJA::Index_type Index_type;
///
typedef Index_type* Index_ptr;


/*!
 ******************************************************************************
 *
 * \brief Integer types used in kernels.
 *
 ******************************************************************************
 */
typedef int Int_type;
///
typedef Int_type* Int_ptr;


/*!
 ******************************************************************************
 *
 * \brief Type used for all kernel checksums.
 *
 ******************************************************************************
 */
typedef long double Checksum_type;


/*!
 ******************************************************************************
 *
 * \brief Floating point types used in kernels.
 *
 ******************************************************************************
 */
#if defined(RP_USE_DOUBLE)
///
typedef double Real_type;

#elif defined(RP_USE_FLOAT)
///
typedef float Real_type;

#else
#error Real_type is undefined!

#endif

typedef Real_type* Real_ptr;
typedef Real_type* RAJA_RESTRICT ResReal_ptr;


#if defined(RP_USE_COMPLEX)
///
typedef std::complex<Real_type> Complex_type;

typedef Complex_type* Complex_ptr;
typedef Complex_type* RAJA_RESTRICT ResComplex_ptr;
#endif


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
