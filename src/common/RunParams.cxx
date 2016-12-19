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
   length_fraction(1.0),
   run_kernels("all"),
   run_variants("all"), 
   output_file_prefix("RAJA_Perf_Suite")
{
  for (int i = 1; i < argc; ++i) {

    if ( std::string(argv[i]) == std::string("--npasses") ) {

      npasses = ::atoi( argv[++i] );

    } else if ( std::string(argv[i]) == std::string("--sampfrac") ) {

      sample_fraction = ::atof( argv[++i] );

    } else if ( std::string(argv[i]) == std::string("--lenfrac") ) {

      length_fraction = ::atof( argv[++i] );

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

    } else if ( std::string(argv[i]) == std::string("--help") ) {

      std::cout << "\n\n";
      std::cout << "Usage: ./raja-perf [options] ";
// RDH describe options...
      std::cout << std::endl;
      std::cout.flush();

    }

  }
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

}  // closing brace for rajaperf namespace
