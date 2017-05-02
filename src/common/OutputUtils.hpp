/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for utility routines for output generation.
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

#ifndef OutputUtils_HXX
#define OutputUtils_HXX

#include <string>
#include<iostream>
#include<iomanip>
#include<sstream>
#include<fstream>

namespace rajaperf
{

/*!
 * \brief Recursively construct directories based on a relative or 
 * absolute path name.  
 * 
 * Return string name of directory if created successfully, else empty string.
 */
std::string recursiveMkdir(const std::string& in_path);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
