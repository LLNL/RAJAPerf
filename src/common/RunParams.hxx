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


#include "RAJAPerfSuite.hxx"

#include <string>
#include <vector>


#ifndef RunParams_HXX
#define RunParams_HXX

namespace rajaperf
{

class KernelBase;

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

  int npasses;                     /*!< Number of passes through suite.  */
  double sample_fraction;          /*!< Frac of default kernel samples to run */
  double length_fraction;          /*!< Frac of default kernel length to run */

  std::string run_kernels;         /*!< Filter which kernels to run... */
  std::string run_variants;        /*!< Filter which variants to run... */

  std::string output_file_prefix;  /*!< Prefix for output data file. */

  std::vector<KernelBase*> kernels;/*!< Vector of kernel objects to run */
  std::vector<VariantID> variants; /*!< Vector of variant IDs to run */

private:
  RunParams() = delete;

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
