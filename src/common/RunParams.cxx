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

#include "RAJAPerfSuite.hxx"

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
 : good2go(true),
   npasses(1),
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
  for (int i = 1; i < argc && good2go; ++i) {

    std::string opt(std::string(argv[i]));

    if ( opt == std::string("--help") ) {

      printHelpMessage(std::cout);

      good2go = false;

    } else if ( opt == std::string("--print-kernels") ) {
     
      printKernelNames(std::cout);     

      good2go = false;
 
    } else if ( opt == std::string("--print-variants") ) {

      printVariantNames(std::cout);     

      good2go = false;
 
    } else if ( opt == std::string("--print-suites") ) {

      printSuiteNames(std::cout);     

      good2go = false;

    } else if ( opt == std::string("--npasses") ) {

      i++;
      if ( i < argc ) { 
        npasses = ::atoi( argv[i] );
      } else {
        std::cout << "\nBad input: must give --npasses an integer value" 
                  << std::endl; 
        good2go = false;
      }

    } else if ( opt == std::string("--sampfrac") ) {

      i++;
      if ( i < argc ) { 
        sample_fraction = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input: must give --sampfrac a double value" 
                  << std::endl;       
        good2go = false;
      }

    } else if ( opt == std::string("--sizefrac") ) {

      i++;
      if ( i < argc ) { 
        size_fraction = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input: must give --sizefrac a double value"        
                  << std::endl;
        good2go = false;
      }

    } else if ( opt == std::string("--kernels") ) {

#if 0 // RDH TODO FIX THIS
      i++;  
      if ( i < argc ) {
        opt = std::string(argv[i]);

        while ( i < argc && opt.at(0) != '-' ) {
          kernel_filter.push_back(opt);
          if ( i+1 < argc ) {
            opt = std::string(argv[++i]);
          }
        }
      }
#endif

    } else if ( std::string(argv[i]) == std::string("--variants") ) {

#if 0 // RDH TODO FIX THIS
      if ( i+1 < argc ) {
        opt = std::string(argv[++i]);

        while ( i < argc && opt.at(0) != '-' ) {
          variant_filter.push_back(opt);
          opt = std::string(argv[++i]);
        }
      }
#endif

    } else if ( std::string(argv[i]) == std::string("--outfile") ) {

      if ( i+1 < argc ) {
        output_file_prefix = std::string( argv[++i] );
      }

    } else {
     
      std::string huh(argv[i]);   
      std::cout << "\nUnknown option: " << huh << std::endl;
      std::cout.flush();

      good2go = false;

    }

  }
}


void RunParams::printHelpMessage(std::ostream& str) const
{
  str << "\nUsage: ./raja-perf.exe [options]\n";
  str << "Valid options are:\n"; 

  str << "\t --help (prints options with descriptions}\n";
  str << "\t --print-kernels (prints valid kernel names}\n";
  str << "\t --print-variants (prints valid variant names}\n";
  str << "\t --print-suites (prints valid suite names}\n";
  str << "\t --npasses <int>\n"
            << "\t      (num passes through suite)\n"; 
  str << "\t --sampfrac <double>\n"
            << "\t      (fraction of default # times to run each kernel)\n";
  str << "\t --sizefrac <double>\n"
            << "\t      (fraction of default kernel iteration space size to run)\n";
  str << "\t --kernels <space-separated list of strings>\n"
            << "\t      (names of kernels and/or suites to run)\n"; 
  str << "\t\t e.g.,\n"
            << "\t\t Polybench (run all kernels in Polybench suite)\n"
            << "\t\t INIT3 MULADDSUB (run INIT3 and MULADDSUB kernels\n"
            << "\t\t INIT3 Apps (run INIT3 kernsl and all kernels in Apps suite)\n"
            << "\t\t (no string will runn all kernels)\n";
  str << "\t --variants <space-separated list of strings>\n"
            << "\t      (names of variants)\n"; 
  str << "\t\t e.g.,\n"
            << "\t\t Baseline RAJA_CUDA (run Baseline and RAJA_CUDA kernel variants)\n"
            << "\t\t (no string will run all variants)\n";

  str << std::endl;
  str.flush();
}


void RunParams::printKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels:";
  str << "\n------------------\n";
  for (int ik = 0; ik < NumKernels; ++ik) {
    str << getKernelName(static_cast<KernelID>(ik)) << std::endl;
  }
  str.flush();
}


void RunParams::printVariantNames(std::ostream& str) const
{
  str << "\nAvailable variants:";
  str << "\n-------------------\n";
  for (int iv = 0; iv < NumVariants; ++iv) {
    str << getVariantName(static_cast<VariantID>(iv)) << std::endl;
  }
  str.flush();
}


void RunParams::printSuiteNames(std::ostream& str) const
{
  str << "\nAvailable suites:";
  str << "\n-----------------\n";
  for (int is = 0; is < NumSuites; ++is) {
    str << getSuiteName(static_cast<SuiteID>(is)) << std::endl;
  }
  str.flush();
}

}  // closing brace for rajaperf namespace
