/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for RunParams class.
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


#include "RunParams.hxx"

#include <cstdlib>
#include <cstdio>
#include <iostream>

namespace rajaperf
{

/*
 *******************************************************************************
 *
 * Ctor for PunParams class defines suite execution defaults and parses
 * command line args to set others that are specified when suite is run.
 *
 *******************************************************************************
 */
RunParams::RunParams(int argc, char** argv)
 : npasses(1),
   sample_fraction(1.0),
   size_fraction(1.0),
   kernel_filter(),
   variant_filter(),
   output_file_prefix("RAJA_Perf_Suite")
{
  parseCommandLineOptions(argc, argv);
}


/*
 *******************************************************************************
 *
 * Dtor for RunParams class.
 *
 *******************************************************************************
 */
RunParams::~RunParams()
{
}


/*
 *******************************************************************************
 *
 * Parse command line args to set how suite will run.
 *
 *******************************************************************************
 */
void RunParams::parseCommandLineOptions(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i) {

    if ( std::string(argv[i]) == std::string("--help") ) {

      std::cout << "\n\n";
      std::cout << "Usage: ./raja-perf.exe [options] ";
// RDH output formatted description of options and defaults...
/* 
      ./raja-perf.exe \
      --help [print options with descriptions]
      --print-kernels [print list of kernel names]
      --print-variants [print list of variant names]
      --print-suites [print list of suite names]
      --npasses <int num passes through suite> 
      --sampfrac <double fraction of default # times each kernel is run> 
      --sizefrac <double fraction of default kernel iteration space size to run>
      --kernels <list of strings: kernel names and/or suite names> 
                 e.g.,
                 polybench [runs all kernels in polybench suite]
                 INIT3 MULADDSUB [runs INIT3 and MULADDSUB kernels]
                 INIT3 apps [runs INIT3 kernel and all kernels in apps wuite])
      --variants <list of strings: kernel variants>
                 e.g., 
                 BASELINE RAJA_CUDA [runs BASELINE and  RAJA_CUDA variants]
*/
      std::cout << std::endl;
      std::cout.flush();

    } else if ( std::string(argv[i]) == std::string("--print-kernels") ) {
     
      // print list of kernel names
 
    } else if ( std::string(argv[i]) == std::string("--print-variants") ) {
     
      // print list of variant names
 
    } else if ( std::string(argv[i]) == std::string("--print-suites") ) {

      // print list of suite names 

    } else if ( std::string(argv[i]) == std::string("--npasses") ) {

      npasses = ::atoi( argv[++i] );

    } else if ( std::string(argv[i]) == std::string("--sampfrac") ) {

      sample_fraction = ::atof( argv[++i] );

    } else if ( std::string(argv[i]) == std::string("--sizefrac") ) {

      size_fraction = ::atof( argv[++i] );

    } else if ( std::string(argv[i]) == std::string("--kernels") ) {

      // RDH TODO...
      std::cout << "\n\n";
      std::cout << "Kernel filter option not implemented!\n";
      std::cout << "\tRunning all kernels by default..." << std::endl;
      std::cout.flush();

    } else if ( std::string(argv[i]) == std::string("--variants") ) {

      // RDH TODO...
      std::cout << "\n\n";
      std::cout << "Variant filter not implemented!\n";
      std::cout << "\tRunning all variants by default..." << std::endl;
      std::cout.flush();

    } else if ( std::string(argv[i]) == std::string("--outfile") ) {

      output_file_prefix = std::string( argv[++i] );

    } else {
     
      std::string huh(argv[i]);   
      std::cout << "\nUnknown option: " << huh << std::endl;
      std::cout.flush();

    }

  }
}

}  // closing brace for rajaperf namespace
