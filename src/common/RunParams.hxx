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

#ifndef RAJAPerf_RunParams_HXX
#define RAJAPerf_RunParams_HXX

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

  bool goodToRun() const { return good2go; } 

  int getNumPasses() const { return npasses; }

  double getSampleFraction() const { return sample_fraction; }

  double getSizeFraction() const { return size_fraction; }

  const std::vector<std::string>& getKernelFilter() const 
                                  { return kernel_filter; }

  const std::vector<std::string>& getVariantFilter() const 
                                  { return variant_filter; }

private:
  RunParams() = delete;

  void parseCommandLineOptions(int argc, char** argv);

  bool good2go;                    /*!< true if input is valid for run */

  int npasses;                     /*!< Number of passes through suite  */
  double sample_fraction;          /*!< Frac of default kernel samples to run */
  double size_fraction;            /*!< Frac of default kernel iteration space
                                        to run */

  std::vector<std::string> kernel_filter;  /*!< Filter for kernels to run... */
  std::vector<std::string> variant_filter; /*!< Filter for variants to run... */

  std::string output_file_prefix;  /*!< Prefix for output data file. */

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
