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
 : input_state(Undefined),
   npasses(1),
   sample_fraction(1.0),
   size_fraction(1.0),
   kernel_input(),
   unknown_kernel_input(),
   variant_input(),
   unknown_variant_input(),
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
 * Print all run params data to given output stream.
 *
 *******************************************************************************
 */
void RunParams::print(std::ostream& str) const
{
  str << "\n npasses = " << npasses; 
  str << "\n sample_fraction = " << sample_fraction; 
  str << "\n size_fraction = " << size_fraction; 
  str << "\n output_file_prefix = " << output_file_prefix; 

  str << "\n kernel_input = "; 
  for (size_t j = 0; j < kernel_input.size(); ++j) {
    str << "\n\t" << kernel_input[j];
  }
  str << "\n unknown_kernel_input = ";
  for (size_t j = 0; j < unknown_kernel_input.size(); ++j) {
    str << "\n\t" << unknown_kernel_input[j];
  }

  str << "\n variant_input = "; 
  for (size_t j = 0; j < variant_input.size(); ++j) {
    str << "\n\t" << variant_input[j];
  }
  str << "\n unknown_variant_input = "; 
  for (size_t j = 0; j < unknown_variant_input.size(); ++j) {
    str << "\n\t" << unknown_variant_input[j];
  }

  str << std::endl;
  str.flush();
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

    std::string opt(std::string(argv[i]));

    if ( opt == std::string("--help") ||
         opt == std::string("-h") ) {

      printHelpMessage(std::cout);
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-kernels") ||
                opt == std::string("-pk") ) {
     
      printFullKernelNames(std::cout);     
      input_state = InfoRequest;
 
    } else if ( opt == std::string("--print-variants") ||
                opt == std::string("-pv") ) {

      printVariantNames(std::cout);     
      input_state = InfoRequest;
 
    } else if ( opt == std::string("--npasses") ) {

      i++;
      if ( i < argc ) { 
        npasses = ::atoi( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --npasses a value for number of passes (int)" 
                  << std::endl; 
        input_state = BadInput;
      }

    } else if ( opt == std::string("--samplefrac") ) {

      i++;
      if ( i < argc ) { 
        sample_fraction = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --samplefrac a value for sample fraction (double)" 
                  << std::endl;       
        input_state = BadInput;
      }

    } else if ( opt == std::string("--sizefrac") ) {

      i++;
      if ( i < argc ) { 
        size_fraction = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --sizefrac a value for size fraction (double)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--kernels") ||
                opt == std::string("-k") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          kernel_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--variants") ||
                std::string(argv[i]) == std::string("-v") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          variant_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--outfile") ||
                std::string(argv[i]) == std::string("-out") ) {

      if ( i+1 < argc ) {
        output_file_prefix = std::string( argv[++i] );
      }

    } else {
     
      input_state = BadInput;

      std::string huh(argv[i]);   
      std::cout << "\nUnknown option: " << huh << std::endl;
      std::cout.flush();

    }

  }
}


void RunParams::printHelpMessage(std::ostream& str) const
{
  str << "\nUsage: ./raja-perf.exe [options]\n";
  str << "Valid options are:\n"; 

  str << "\t --help, -h (prints options with descriptions}\n";
  str << "\t --print-kernels, -pk (prints valid kernel names}\n";
  str << "\t --print-variants, -pv (prints valid variant names}\n";
  str << "\t --npasses <int>\n"
      << "\t      (num passes through suite)\n"; 
  str << "\t --samplefrac <double>\n"
      << "\t      (fraction of default # times to run each kernel)\n";
  str << "\t --sizefrac <double>\n"
      << "\t      (fraction of default kernel iteration space size to run)\n";
  str << "\t --outfile, -out <string>\n"
      << "\t      (name of output file prefix)\n";
  str << "\t --kernels, -k <space-separated list of strings>\n"
      << "\t      (names of individual kernels and/or groups of kernels to run)\n"; 
  str << "\t\t Examples...\n"
      << "\t\t --kernels Polybench (run all kernels in Polybench group)\n"
      << "\t\t -k INIT3 MULADDSUB (run INIT3 and MULADDSUB kernels\n"
      << "\t\t -k INIT3 Apps (run INIT3 kernsl and all kernels in Apps group)\n"
      << "\t\t (if no string(s) given, all kernels will be run)\n";
  str << "\t --variants, -k <space-separated list of strings>\n"
      << "\t      (names of variants)\n"; 
  str << "\t\t Examples...\n"
      << "\t\t -variants RAJA_CUDA (run RAJA_CUDA variants)\n"
      << "\t\t -v Baseline RAJA_CUDA (run Baseline, RAJA_CUDA variants)\n"
      << "\t\t (if no string(s) given, all variants will be run)\n";

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


void RunParams::printFullKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels (<group name>_<kernel name>):";
  str << "\n-----------------------------------------\n";
  for (int ik = 0; ik < NumKernels; ++ik) {
    str << getFullKernelName(static_cast<KernelID>(ik)) << std::endl;
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


void RunParams::printGroupNames(std::ostream& str) const
{
  str << "\nAvailable groups:";
  str << "\n-----------------\n";
  for (int is = 0; is < NumGroups; ++is) {
    str << getGroupName(static_cast<GroupID>(is)) << std::endl;
  }
  str.flush();
}

}  // closing brace for rajaperf namespace
