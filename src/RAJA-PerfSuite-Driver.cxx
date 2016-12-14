/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file that drives performance suite.
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

#include "common/RAJAPerfSuite.hxx"

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // STEP 0: Parse command line options
  RAJAperf::RunParams params(argc, argv);

  // STEP 1: Report parameter summary
  RAJAperf::reportRunSummary(params);  
    
  // STEP 2: Run the loop suite
  RAJAperf::Executor executor(params);
  executor.run();

  // STEP 3: Write execution reports
  RAJAperf::outputRunData(params);  

  return 0;
}

//------------------------------------------------------------------------------
void parseInputParams(int argc, char** argv)
{
  // Set defaults
  Parameters.npasses             = 1;
  Parameters.run_kernels         = std::string("all");
  Parameters.run_variants        = std::string("all");
  Parameters.length_fraction     = 1.0;
  Parameters.output_file_prefix  = std::string("RAJA_Perf_Suite");

  for (int i = 1; i < argc; ++i) {

    if ( strcmp( argv[i],"--file" )==0 ) {

      Parameters.useFile  = true;
      Parameters.fileName = std::string( argv[++i] );

    } else if ( strcmp( argv[i], "--niters" ) == 0 ) {

      Parameters.niters = atoi( argv[++i] );

    } else if ( strcmp( argv[i], "--nx" )==0 ) {

      Parameters.nx = atoi( argv[++i] );

    } else if ( strcmp( argv[i], "--ny" )==0 ) {

      Parameters.ny = atoi( argv[++i] );

    } else if ( strcmp( argv[i], "--nz" )==0 ) {

      Parameters.nz = atoi( argv[++i] );

    } else if ( strcmp( argv[i],"--ngleft" )==0 ) {

      Parameters.ngleft = atoi( argv[++i] );

    } else if ( strcmp( argv[i],"--ngright" )==0 ) {

      Parameters.ngright = atoi( argv[++i] );

    } else if ( strcmp( argv[i], "--nreg" )==0 ) {

      Parameters.nreg = atoi( argv[++i] );

    } else if ( strcmp( argv[i], "--islab")==0 ) {

      Parameters.mode = loopsuite::ISLAB;

    } else if ( strcmp( argv[i], "--jslab" )==0 ) {

      Parameters.mode = loopsuite::JSLAB;

    } else if ( strcmp(argv[i],"--kslab")==0 ) {

      Parameters.mode = loopsuite::KSLAB;

    } else if ( strcmp(argv[i],"--sphere")==0 ) {

      Parameters.mode = loopsuite::SPHERICAL;

    } else if ( strcmp(argv[i],"--filter")==0 ) {

      Parameters.keyword = std::string( argv[++i] );


    } else if ( strcmp(argv[i],"--vtk")==0 ) {

      Parameters.dumpVtk = true;
      Parameters.vtkFile = std::string( argv[++i] );

    } else if ( strcmp(argv[i],"--mixed-zones")==0 ) {

      Parameters.mixed_zones = true;

    } else if ( strcmp(argv[i], "--help")==0 ) {

      std::cout << "\n\n";
      std::cout << "Usage: ./loopsuite-driver [--filter <keyword>] ";
      std::cout << " [--file <file>] ";
      std::cout << "or [--nx <Nx> --ny <Ny> --nz <Nz> --nreg <N> [mode] [--mixed-zones] ]" << "\n\n";

      std::cout << "--filter  \truns loop kernels that match a given keyword.";
      std::cout << " Runs all loop kernels by default.\n\n";

      std::cout << "--vtk <file> \t dumps input to a VTK file for visualizing it\n\n.";

      std::cout << "INITIALIZE TEST DATA OUT-OF-CORE\n";
      std::cout << "================================\n";
      std::cout << "--file    \tspecifies the file used to initialize data.";
      std::cout << "\n\n";

      std::cout << "GENERATE SYNTHETIC DATA IN-CORE\n";
      std::cout << "================================\n";
      std::cout << "--nx      \tnumber of grid points along x.\n";
      std::cout << "--ny      \tnumber of grid points along y.\n";
      std::cout << "--nz      \tnumber of grid points along z.\n";
      std::cout << "--nreg    \tnumber of material regions.Default is 2.\n\n";
      std::cout << "[mode]    \tspecifies the material layout to use. ";
      std::cout << "Possible values are the following:\n";
      std::cout << "--islab   \tsplits the domain in 'nreg' slabs along i.\n";
      std::cout << "--jslab   \tsplits the domain in 'nreg' slabs along j.\n";
      std::cout << "--kslab   \tsplits the domain in 'nreg' slabs alonk k.\n";
      std::cout << "--sphere  \tputs 'nreg' material in concentric circles from";
      std::cout << " the logical min.\n\n";
      std::cout << "--mixed-zones \tflag used to generate mixed zones\n";
      std::cout << std::endl;
      std::cout.flush();

      exit(0);
    }

  } // END for all parameters

}
