//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Basic data types used in the Suite.
///

#ifndef RAJAPerf_RPTypes_HPP
#define RAJAPerf_RPTypes_HPP

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
 * \brief Type used for indexing in all kernel repetition loops.
 *
 * It is volatile to ensure that kernels will not be optimized away by 
 * compilers, which can happen in some circumstances.
 *
 ******************************************************************************
 */
typedef volatile int RepIndex_type;


/*!
 ******************************************************************************
 *
 * \brief Types used for all kernel loop indexing.
 *
 ******************************************************************************
 */
#if 0 // Index_type
typedef RAJA::Index_type Index_type;
#else
typedef int Index_type;
#endif
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
