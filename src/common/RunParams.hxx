/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RunParams class.
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

#ifndef RAJAPerfRunParams_HXX
#define RAJAPerfRunParams_HXX

#include <string>
#include <vector>

namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Simple class to hold suite execution parameters.
 *
 *******************************************************************************
 */
class RunParams {
public:
  RunParams( int argc, char** argv );
  ~RunParams( );

  int npasses;                     /*!< Number of passes through suite  */
  double sample_fraction;          /*!< Frac of default kernel samples to run */
  double size_fraction;            /*!< Frac of default kernel iteration space
                                        to run */

  std::vector<std::string> kernel_filter;  /*!< Filter for kernels to run... */
  std::vector<std::string> variant_filter; /*!< Filter for variants to run... */

  std::string output_file_prefix;  /*!< Prefix for output data file. */

private:
  RunParams() = delete;

  void parseCommandLineOptions(int argc, char** argv);

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
