//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RunParams.hpp"

#include "KernelBase.hpp"

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
   show_progress(false),
   npasses(1),
   npasses_combiners(),
   rep_fact(1.0),
   size_meaning(SizeMeaning::Unset),
   size(0.0),
   size_factor(0.0),
   data_alignment(RAJA::DATA_ALIGN),
   gpu_stream(1),
   gpu_block_sizes(),
   pf_tol(0.1),
   checkrun_reps(1),
   reference_variant(),
   kernel_input(),
   invalid_kernel_input(),
   exclude_kernel_input(),
   invalid_exclude_kernel_input(),
   variant_input(),
   invalid_variant_input(),
   exclude_variant_input(),
   invalid_exclude_variant_input(),
   tuning_input(),
   invalid_tuning_input(),
   exclude_tuning_input(),
   invalid_exclude_tuning_input(),
   feature_input(),
   invalid_feature_input(),
   exclude_feature_input(),
   invalid_exclude_feature_input(),
   npasses_combiner_input(),
   invalid_npasses_combiner_input(),
   outdir(),
   outfile_prefix("RAJAPerf"),
   disable_warmup(false)
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
  str << "\n show_progress = " << show_progress;
  str << "\n npasses = " << npasses;
  str << "\n npasses combiners = ";
  for (size_t j = 0; j < npasses_combiners.size(); ++j) {
    str << "\n\t" << CombinerOptToStr(npasses_combiners[j]);
  }
  str << "\n npasses_combiners_input = ";
  for (size_t j = 0; j < npasses_combiner_input.size(); ++j) {
    str << "\n\t" << npasses_combiner_input[j];
  }
  str << "\n invalid_npasses_combiners_input = ";
  for (size_t j = 0; j < invalid_npasses_combiner_input.size(); ++j) {
    str << "\n\t" << invalid_npasses_combiner_input[j];
  }
  str << "\n rep_fact = " << rep_fact;
  str << "\n size_meaning = " << SizeMeaningToStr(getSizeMeaning());
  str << "\n size = " << size;
  str << "\n size_factor = " << size_factor;
  str << "\n data_alignment = " << data_alignment;
  str << "\n gpu stream = " << ((gpu_stream == 0) ? "0" : "RAJA default");
  str << "\n gpu_block_sizes = ";
  for (size_t j = 0; j < gpu_block_sizes.size(); ++j) {
    str << "\n\t" << gpu_block_sizes[j];
  }
  str << "\n pf_tol = " << pf_tol;
  str << "\n checkrun_reps = " << checkrun_reps;
  str << "\n reference_variant = " << reference_variant;
  str << "\n outdir = " << outdir;
  str << "\n outfile_prefix = " << outfile_prefix;

  str << "\n disable_warmup = " << disable_warmup;

  str << "\n seq data space = " << getDataSpaceName(seqDataSpace);
  str << "\n omp data space = " << getDataSpaceName(ompDataSpace);
  str << "\n omp target data space = " << getDataSpaceName(ompTargetDataSpace);
  str << "\n cuda data space = " << getDataSpaceName(cudaDataSpace);
  str << "\n hip data space = " << getDataSpaceName(hipDataSpace);
  str << "\n kokkos data space = " << getDataSpaceName(kokkosDataSpace);

  str << "\n kernel_input = ";
  for (size_t j = 0; j < kernel_input.size(); ++j) {
    str << "\n\t" << kernel_input[j];
  }
  str << "\n invalid_kernel_input = ";
  for (size_t j = 0; j < invalid_kernel_input.size(); ++j) {
    str << "\n\t" << invalid_kernel_input[j];
  }

  str << "\n exclude_kernel_input = ";
  for (size_t j = 0; j < exclude_kernel_input.size(); ++j) {
    str << "\n\t" << exclude_kernel_input[j];
  }
  str << "\n invalid_exclude_kernel_input = ";
  for (size_t j = 0; j < invalid_exclude_kernel_input.size(); ++j) {
    str << "\n\t" << invalid_exclude_kernel_input[j];
  }

  str << "\n variant_input = ";
  for (size_t j = 0; j < variant_input.size(); ++j) {
    str << "\n\t" << variant_input[j];
  }
  str << "\n invalid_variant_input = ";
  for (size_t j = 0; j < invalid_variant_input.size(); ++j) {
    str << "\n\t" << invalid_variant_input[j];
  }

  str << "\n exclude_variant_input = ";
  for (size_t j = 0; j < exclude_variant_input.size(); ++j) {
    str << "\n\t" << exclude_variant_input[j];
  }
  str << "\n invalid_exclude_variant_input = ";
  for (size_t j = 0; j < invalid_exclude_variant_input.size(); ++j) {
    str << "\n\t" << invalid_exclude_variant_input[j];
  }

  str << "\n tuning_input = ";
  for (size_t j = 0; j < tuning_input.size(); ++j) {
    str << "\n\t" << tuning_input[j];
  }
  str << "\n invalid_tuning_input = ";
  for (size_t j = 0; j < invalid_tuning_input.size(); ++j) {
    str << "\n\t" << invalid_tuning_input[j];
  }

  str << "\n exclude_tuning_input = ";
  for (size_t j = 0; j < exclude_tuning_input.size(); ++j) {
    str << "\n\t" << exclude_tuning_input[j];
  }
  str << "\n invalid_exclude_tuning_input = ";
  for (size_t j = 0; j < invalid_exclude_tuning_input.size(); ++j) {
    str << "\n\t" << invalid_exclude_tuning_input[j];
  }

  str << "\n feature_input = ";
  for (size_t j = 0; j < feature_input.size(); ++j) {
    str << "\n\t" << feature_input[j];
  }
  str << "\n invalid_feature_input = ";
  for (size_t j = 0; j < invalid_feature_input.size(); ++j) {
    str << "\n\t" << invalid_feature_input[j];
  }

  str << "\n exclude_feature_input = ";
  for (size_t j = 0; j < exclude_feature_input.size(); ++j) {
    str << "\n\t" << exclude_feature_input[j];
  }
  str << "\n invalid_exclude_feature_input = ";
  for (size_t j = 0; j < invalid_exclude_feature_input.size(); ++j) {
    str << "\n\t" << invalid_exclude_feature_input[j];
  }

  str << std::endl;
  str.flush();
}


/*
 *******************************************************************************
 *
 * Parse command line args to set how suite will run.
 *
 * Important: Some input is checked for correctness in this method. Otherwise,
 *            it is checked for correctness in Executor::setupSuite().
 *
 *******************************************************************************
 */
void RunParams::parseCommandLineOptions(int argc, char** argv)
{
  getCout() << "\n\nReading command line input..." << std::endl;

  for (int i = 1; i < argc; ++i) {

    std::string opt(argv[i]);

    if ( opt == std::string("--help") ||
         opt == std::string("-h") ) {

      printHelpMessage(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--show-progress") ||
                opt == std::string("-sp") ) {

      show_progress = true;

    } else if ( opt == std::string("--print-kernels") ||
                opt == std::string("-pk") ) {

      printFullKernelNames(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-variants") ||
                opt == std::string("-pv") ) {

      printVariantNames(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-data-spaces") ||
                opt == std::string("-pds") ) {

      printDataSpaceNames(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-features") ||
                opt == std::string("-pf") ) {

      printFeatureNames(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-feature-kernels") ||
                opt == std::string("-pfk") ) {

      printFeatureKernels(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--print-kernel-features") ||
                opt == std::string("-pkf") ) {

      printKernelFeatures(getCout());
      input_state = InfoRequest;

    } else if ( opt == std::string("--npasses") ) {

      i++;
      if ( i < argc ) {
        npasses = ::atoi( argv[i] );
      } else {
        getCout() << "\nBad input:"
                  << " must give --npasses a value for number of passes (int)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--npasses-combiners") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          npasses_combiner_input.push_back(opt);
          ++i;
        }
      }

    } else if ( opt == std::string("--repfact") ) {

      i++;
      if ( i < argc ) {
        rep_fact = ::atof( argv[i] );
      } else {
        getCout() << "\nBad input:"
                  << " must give --rep_fact a value (double)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--sizefact") ) {

      i++;
      if ( i < argc ) {
        if (size_meaning == SizeMeaning::Direct) {
          getCout() << "\nBad input:"
                    << " may only set one of --size and --sizefact"
                    << std::endl;
          input_state = BadInput;
        } else {
          size_factor = ::atof( argv[i] );
          if ( size_factor >= 0.0 ) {
            size_meaning = SizeMeaning::Factor;
          } else {
            getCout() << "\nBad input:"
                  << " must give --sizefact a POSITIVE value (double)"
                  << std::endl;
            input_state = BadInput;
          }
        }
      } else {
        getCout() << "\nBad input:"
                  << " must give --sizefact a value (double)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--size") ) {

      i++;
      if ( i < argc ) {
        if (size_meaning == SizeMeaning::Factor) {
          getCout() << "\nBad input:"
                    << " may only set one of --size and --sizefact"
                    << std::endl;
          input_state = BadInput;
        } else {
          size = ::atof( argv[i] );
          if ( size >= 0.0 ) {
            size_meaning = SizeMeaning::Direct;
          } else {
            getCout() << "\nBad input:"
                  << " must give --size a POSITIVE value (double)"
                  << std::endl;
            input_state = BadInput;
          }
        }
      } else {
        getCout() << "\nBad input:"
                  << " must give --size a value (int)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("-align") ||
                opt == std::string("--data_alignment") ) {

      i++;
      if ( i < argc ) {
        long long align = ::atoll( argv[i] );
        long long min_align = alignof(std::max_align_t);
        if ( align < min_align ) {
          getCout() << "\nBad input:"
                << " must give " << opt << " a value of at least " << min_align
                << std::endl;
          input_state = BadInput;
        } else if ( (align & (align-1)) != 0 ) {
          getCout() << "\nBad input:"
                << " must give " << opt << " a power of 2"
                << std::endl;
          input_state = BadInput;
        } else {
          data_alignment = align;
        }
      } else {
        getCout() << "\nBad input:"
                  << " must give " << opt << " a value (int)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--gpu_stream_0") ) {

      gpu_stream = 0;

    } else if ( opt == std::string("--gpu_block_size") ) {

      bool got_someting = false;
      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          got_someting = true;
          int gpu_block_size = ::atoi( opt.c_str() );
          if ( gpu_block_size <= 0 ) {
            getCout() << "\nBad input:"
                      << " must give --gpu_block_size POSITIVE values (int)"
                      << std::endl;
            input_state = BadInput;
          } else {
            gpu_block_sizes.push_back(gpu_block_size);
          }
          ++i;
        }
      }
      if (!got_someting) {
        getCout() << "\nBad input:"
                  << " must give --gpu_block_size one or more values (int)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--pass-fail-tol") ||
                opt == std::string("-pftol") ) {

      i++;
      if ( i < argc ) {
        pf_tol = ::atof( argv[i] );
      } else {
        getCout() << "\nBad input:"
                  << " must give --pass-fail-tol (or -pftol) a value (double)"
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

    } else if ( opt == std::string("--exclude-kernels") ||
                opt == std::string("-ek") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          exclude_kernel_input.push_back(opt);
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

    } else if ( std::string(argv[i]) == std::string("--exclude-variants") ||
                std::string(argv[i]) == std::string("-ev") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          exclude_variant_input.push_back(opt);
          ++i;
        }
      }

    } else if ( opt == std::string("--seq-data-space") ||
                opt == std::string("-sds") ||
                opt == std::string("--omp-data-space") ||
                opt == std::string("-ods") ||
                opt == std::string("--omptarget-data-space") ||
                opt == std::string("-otds") ||
                opt == std::string("--cuda-data-space") ||
                opt == std::string("-cds") ||
                opt == std::string("--hip-data-space") ||
                opt == std::string("-hds") ||
                opt == std::string("--kokkos-data-space") ||
                opt == std::string("-kds") ) {

      bool got_someting = false;
      bool got_something_available = false;
      i++;
      if ( i < argc ) {
        auto opt_name = std::move(opt);
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          for (int ids = 0; ids < static_cast<int>(DataSpace::NumSpaces); ++ids) {
            DataSpace ds = static_cast<DataSpace>(ids);
            if (getDataSpaceName(ds) == opt) {
              got_someting = true;
              got_something_available = isDataSpaceAvailable(ds);
              if (        opt_name == std::string("--seq-data-space") ||
                          opt_name == std::string("-sds") ) {
                seqDataSpace = ds;
              } else if ( opt_name == std::string("--omp-data-space") ||
                          opt_name == std::string("-ods") ) {
                ompDataSpace = ds;
              } else if ( opt_name == std::string("--omptarget-data-space") ||
                          opt_name == std::string("-otds") ) {
                ompTargetDataSpace = ds;
              } else if ( opt_name == std::string("--cuda-data-space") ||
                          opt_name == std::string("-cds") ) {
                cudaDataSpace = ds;
              } else if ( opt_name == std::string("--hip-data-space") ||
                          opt_name == std::string("-hds") ) {
                hipDataSpace = ds;
              } else if ( opt_name == std::string("--kokkos-data-space") ||
                          opt_name == std::string("-kds") ) {
                kokkosDataSpace = ds;
              } else {
                got_someting = false;
              }

              break;
            }
          }
          if (!got_someting) {
            getCout() << "\nBad input:"
                      << " must give " << opt_name << " a valid data space"
                      << std::endl;
            input_state = BadInput;
          } else if (!got_something_available) {
            getCout() << "\nBad input:"
                      << " must give " << opt_name << " a data space this is available in this config"
                      << std::endl;
            input_state = BadInput;
          }
        }
      }
    } else if ( std::string(argv[i]) == std::string("--tunings") ||
                std::string(argv[i]) == std::string("-t") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          tuning_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--exclude-tunings") ||
                std::string(argv[i]) == std::string("-et") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          exclude_tuning_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--features") ||
                std::string(argv[i]) == std::string("-f") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          feature_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--exclude-features") ||
                std::string(argv[i]) == std::string("-ef") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          exclude_feature_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--outdir") ||
                std::string(argv[i]) == std::string("-od") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          outdir = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--outfile") ||
                std::string(argv[i]) == std::string("-of") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          outfile_prefix = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--refvar") ||
                std::string(argv[i]) == std::string("-rv") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          reference_variant = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--dryrun") ) {

      if (input_state != BadInput) {
        input_state = DryRun;
      }

    } else if ( std::string(argv[i]) == std::string("--disable-warmup") ) {

      disable_warmup = true;

    } else if ( std::string(argv[i]) == std::string("--checkrun") ) {

      input_state = CheckRun;

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          int cr = ::atoi( argv[i] );
          if ( cr < 0 ) {
            getCout() << "\nBad input:"
                      << " must give --checkrun a non-negative value (int)"
                      << std::endl;
            input_state = BadInput;
          } else {
            checkrun_reps = cr;
          }
        }

      }

    } else {

      input_state = BadInput;

      std::string huh(argv[i]);
      getCout() << "\nUnknown option: " << huh << std::endl;
      getCout().flush();

    }

  }

  // Default size and size_meaning if unset
  if (size_meaning == SizeMeaning::Unset) {
    size_meaning = SizeMeaning::Factor;
    size_factor = 1.0;
  }

  // Default npasses_combiners if no input
  if (npasses_combiner_input.empty()) {
    npasses_combiners.emplace_back(CombinerOpt::Average);
  }
}


void RunParams::printHelpMessage(std::ostream& str) const
{
  str << "\nUsage: ./raja-perf.exe [options]\n";

  str << "Valid options are:\n";

  str << "\t --help, -h (print available command line options with descriptions)\n\n";

  str << "\t Print info about option arguments....\n"
      << "\t =====================================\n\n";;

  str << "\t --print-kernels, -pk (print names of available kernels to run)\n\n";

  str << "\t --print-variants, -pv (print names of available variants to run)\n\n";

  str << "\t --print-features, -pf (print names of RAJA features exercised in Suite)\n\n";

  str << "\t --print-feature-kernels, -pfk \n"
      << "\t      (print names of kernels that use each feature)\n\n";

  str << "\t --print-kernel-features, -pkf \n"
      << "\t      (print names of features used by each kernel)\n\n";

  str << "\t --print-data-spaces, -pds (print names of data spaces)\n\n";

  str << "\t Selecting output details....\n"
      << "\t ============================\n\n";;

  str << "\t --show-progress, -sp (print execution progress during run)\n\n";

  str << "\t --dryrun (print summary of how Suite will run without running it)\n\n";

  str << "\t --refvar, -rv <string> [Default is none]\n"
      << "\t      (reference variant for speedup calculation)\n\n";
  str << "\t\t Example...\n"
      << "\t\t --refvar Base_Seq (speedups reported relative to Base_Seq variants)\n\n";

  str << "\t --pass-fail-tol, -pftol <double> [default is 0.1; i.e., 10%]\n"
      << "\t      (slowdown tolerance for RAJA vs. Base variants in FOM report)\n";
  str << "\t\t Example...\n"
      << "\t\t -pftol 0.2 (RAJA kernel variants that run 20% or more slower than Base variants will be reported as OVER_TOL in FOM report)\n\n";

  str << "\t --outdir, -od <string> [Default is current directory]\n"
      << "\t      (directory path for output data files)\n";
  str << "\t\t Examples...\n"
      << "\t\t --outdir foo (output files to ./foo directory\n"
      << "\t\t -od /nfs/tmp/me (output files to /nfs/tmp/me directory)\n\n";

  str << "\t --outfile, -of <string> [Default is RAJAPerf]\n"
      << "\t      (file name prefix for output files)\n";
  str << "\t\t Examples...\n"
      << "\t\t --outfile mydata (output data will be in files 'mydata*')\n"
      << "\t\t -of dat (output data will be in files 'dat*')\n\n";

  str << "\t Selecting kernels to run....\n"
      << "\t ==========================\n\n";;

  str << "\t --disable-warmup (disable warmup kernel) [Default is run all warmup kernels]\n\n";

  str << "\t --kernels, -k <space-separated strings> [Default is run all]\n"
      << "\t      (names of individual kernels and/or groups of kernels to run)\n"
      << "\t      See '--print-kernels'/'-pk' option for list of valid kernel and group names.\n"
      << "\t      Kernel names are listed as <group name>_<kernel name>.\n";
  str << "\t\t Examples...\n"
      << "\t\t --kernels Polybench (run all kernels in Polybench group)\n"
      << "\t\t -k INIT3 MULADDSUB (run INIT3 and MULADDSUB kernels)\n"
      << "\t\t -k INIT3 Apps (run INIT3 kernel and all kernels in Apps group)\n\n";

  str << "\t --exclude-kernels, -ek <space-separated strings> [Default is exclude none]\n"
      << "\t      (names of individual kernels and/or groups of kernels to exclude)\n"
      << "\t      See '--print-kernels'/'-pk' option for list of valid kernel and group names.\n"
      << "\t      Kernel names are listed as <group name>_<kernel name>.\n";
  str << "\t\t Examples...\n"
      << "\t\t --exclude-kernels Polybench (exclude all kernels in Polybench group)\n"
      << "\t\t -ek INIT3 MULADDSUB (exclude INIT3 and MULADDSUB kernels)\n"
      << "\t\t -ek INIT3 Apps (exclude INIT3 kernel and all kernels in Apps group)\n\n";

  str << "\t --variants, -v <space-separated strings> [Default is run all]\n"
      << "\t      (names of variants to run)\n"
      << "\t      See '--print-variants'/'-pv' option for list of valid variant names.\n";
  str << "\t\t Examples...\n"
      << "\t\t --variants RAJA_CUDA (run all RAJA_CUDA kernel variants)\n"
      << "\t\t -v Base_Seq RAJA_CUDA (run Base_Seq and  RAJA_CUDA variants)\n\n";

  str << "\t --exclude-variants, -ev <space-separated strings> [Default is exclude none]\n"
      << "\t      (names of variants to exclude)\n"
      << "\t      See '--print-variants'/'-pv' option for list of valid variant names.\n";
  str << "\t\t Examples...\n"
      << "\t\t --exclude-variants RAJA_CUDA (exclude all RAJA_CUDA kernel variants)\n"
      << "\t\t -ev Base_Seq RAJA_CUDA (exclude Base_Seq and  RAJA_CUDA variants)\n\n";

  str << "\t --features, -f <space-separated strings> [Default is run all]\n"
      << "\t      (names of features to run)\n"
      << "\t      See '--print-kernel-features'/'-pkf' option for list of RAJA features used by kernels.\n";
  str << "\t\t Examples...\n"
      << "\t\t --features Forall (run all kernels that use RAJA forall)\n"
      << "\t\t -f Forall Reduction (run all kernels that use RAJA forall or RAJA reductions)\n\n";

  str << "\t --exclude-features, -ef <space-separated strings> [Default is exclude none]\n"
      << "\t      (names of features to exclude)\n"
      << "\t      See '--print-kernel-features'/'-pkf' option for list of RAJA features used by kernels.\n";
  str << "\t\t Examples...\n"
      << "\t\t --exclude-features Forall (exclude all kernels that use RAJA forall)\n"
      << "\t\t -ef Forall Reduction (exclude all kernels that use RAJA forall or RAJA reductions)\n\n";

  str << "\t Selecting run size....\n"
      << "\t ======================\n\n";;

  str << "\t --checkrun <int> [default is 1]\n"
      << "\t      (run each kernel a given number of times)\n"
      << "\t      Helpful to check things are working properly or\n" 
      << "\t      run a small sample to reduce aggregate execution time)\n";
  str << "\t\t Example...\n"
      << "\t\t --checkrun 2 (run each kernel twice)\n\n";

  str << "\t --npasses <int> [default is 1]\n"
      << "\t      (num passes through Suite)\n";
  str << "\t\t Example...\n"
      << "\t\t --npasses 2 (runs complete Suite twice)\n\n";

  str << "\t --npasses-combiners <space-separated strings> [Default is 'Average']\n"
      << "\t      (Specify combining npasses timing data into timing files)\n";
  str << "\t\t Example...\n"
      << "\t\t --npasses-combiners Average Minimum Maximum (produce average, min, and\n"
      << "\t\t   max timing .csv files)\n\n";

  str << "\t --repfact <double> [default is 1.0]\n"
      << "\t      (multiplier on default # reps to run each kernel)\n";
  str << "\t\t Example...\n"
      << "\t\t --repfact 0.5 (run each kernels 1/2 as many times as its default reps)\n\n";

  str << "\t --sizefact <double> [default is 1.0]\n"
      << "\t      (fraction of default kernel sizes to run)\n"
      << "\t      May not be set if '--size' is set.\n";
  str << "\t\t Example...\n"
      << "\t\t --sizefact 2.0 (run each kernel with size twice its default size)\n\n";

  str << "\t --size <int> [no default]\n"
      << "\t      (kernel size to run for all kernels)\n"
      << "\t      May not be set if --sizefact is set.\n";
  str << "\t\t Example...\n"
      << "\t\t --size 1000000 (runs each kernel with size ~1,000,000)\n\n";

  str << "\t Selecting GPU execution details....\n"
      << "\t ===================================\n\n";;

  str << "\t --gpu_stream_0 [default is to use RAJA default stream]\n"
      << "\t      (when this option is given, use stream 0 with HIP and CUDA kernel variants)\n\n";

  str << "\t --gpu_block_size <space-separated ints> [no default]\n"
      << "\t      (block sizes to run for all GPU kernels)\n"
      << "\t      GPU kernels not supporting gpu_block_size option will be skipped.\n"
      << "\t      Behavior depends on kernel implementations and \n"
      << "\t      values give via CMake variable RAJA_PERFSUITE_GPU_BLOCKSIZES.\n";
  str << "\t\t Example...\n"
      << "\t\t --gpu_block_size 128 256 512 (runs kernels with gpu_block_size 128, 256, and 512)\n\n";

  str << "\t --tunings, -t <space-separated strings> [Default is run all]\n"
      << "\t      (names of tunings to run)\n"
      << "\t      Note: knowing which tunings are available requires knowledge about the variants,\n"
      << "\t      since available tunings depend on the given variant (and potentially other args).\n";
  str << "\t\t Examples...\n"
      << "\t\t --tunings default (run all default tunings)\n"
      << "\t\t -t default block_128 (run default and block_128 tunings)\n\n";

  str << "\t --exclude-tunings, -et <space-separated strings> [Default is exclude none]\n"
      << "\t      (names of tunings to exclude)\n"
      << "\t      See --tunings option for more information.\n";
  str << "\t\t Examples...\n"
      << "\t\t --exclude-tunings library (exclude all library tunings)\n"
      << "\t\t -et default library (exclude default and library tunings)\n\n";

  str << "\t Selecting kernel data details....\n"
      << "\t =================================\n\n";;

  str << "\t --data_alignment, -align <int> [default is RAJA::DATA_ALIGN]\n"
      << "\t      (minimum memory alignment for host allocations)\n"
      << "\t      Must be a power of 2 at least as large as default alignment.\n";
  str << "\t\t Example...\n"
      << "\t\t -align 4096 (allocates memory aligned to 4KiB boundaries)\n\n";

  str << "\t --seq-data-space, -sds <string> [Default is Host]\n"
      << "\t      (name of data space to use for sequential variants)\n"
      << "\t      Valid data space names are 'Host' or 'CudaPinned'\n";
  str << "\t\t Examples...\n"
      << "\t\t --seq-data-space Host (run sequential variants with Host memory)\n"
      << "\t\t -sds CudaPinned (run sequential variants with Cuda Pinned memory)\n\n";

  str << "\t --omp-data-space, -ods <string> [Default is Omp]\n"
      << "\t      (names of data space to use for OpenMP variants)\n"
      << "\t      Valid data space names are 'Host' or 'Omp'\n";
  str << "\t\t Examples...\n"
      << "\t\t --omp-data-space Omp (run Omp variants with Omp memory)\n"
      << "\t\t -ods Host (run Omp variants with Host memory)\n\n";

  str << "\t --omptarget-data-space, -otds <string> [Default is OmpTarget]\n"
      << "\t      (names of data space to use for OpenMP Target variants)\n"
      << "\t      Valid data space names are 'OmpTarget' or 'CudaPinned'\n";
  str << "\t\t Examples...\n"
      << "\t\t --omptarget-data-space OmpTarget (run Omp Target variants with Omp Target memory)\n"
      << "\t\t -otds CudaPinned (run Omp Target variants with Cuda Pinned memory)\n\n";

  str << "\t --cuda-data-space, -cds <string> [Default is CudaDevice]\n"
      << "\t      (names of data space to use for CUDA variants)\n"
      << "\t      Valid data space names are 'CudaDevice', 'CudaPinned', or 'CudaManaged'\n";
  str << "\t\t Examples...\n"
      << "\t\t --cuda-data-space CudaManaged (run CUDA variants with Cuda Managed memory)\n"
      << "\t\t -cds CudaPinned (run CUDA variants with Cuda Pinned memory)\n\n";

  str << "\t --hip-data-space, -hds <string> [Default is HipDevice]\n"
      << "\t      (names of data space to use for HIP variants)\n"
      << "\t      Valid data space names are 'HipDevice', 'HipPinned', or 'HipManaged'\n";
  str << "\t\t Examples...\n"
      << "\t\t --hip-data-space HipManaged (run HIP variants with Hip Managed memory)\n"
      << "\t\t -hds HipPinned (run HIP variants with Hip Pinned memory)\n\n";

  str << "\t --kokkos-data-space, -kds <string> [Default is Host]\n"
      << "\t      (names of data space to use)\n";
  str << "\t\t Examples...\n"
      << "\t\t --kokkos-data-space Host (run KOKKOS variants with Host memory)\n"
      << "\t\t -kds HipPinned (run KOKKOS variants with Hip Pinned memory)\n\n";

  str << std::endl;
  str.flush();
}


void RunParams::printKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels:";
  str << "\n------------------\n";
  for (int kid = 0; kid < NumKernels; ++kid) {
    str << getKernelName(static_cast<KernelID>(kid)) << std::endl;
  }
  str.flush();
}


void RunParams::printFullKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels (<group name>_<kernel name>):";
  str << "\n-----------------------------------------\n";
  for (int kid = 0; kid < NumKernels; ++kid) {
    str << getFullKernelName(static_cast<KernelID>(kid)) << std::endl;
  }
  str.flush();
}


void RunParams::printVariantNames(std::ostream& str) const
{
  str << "\nAvailable variants:";
  str << "\n-------------------\n";
  for (int vid = 0; vid < NumVariants; ++vid) {
    str << getVariantName(static_cast<VariantID>(vid)) << std::endl;
  }
  str.flush();
}


void RunParams::printDataSpaceNames(std::ostream& str) const
{
  str << "\nAvailable data spaces:";
  str << "\n-------------------\n";
  for (int ids = 0; ids < static_cast<int>(DataSpace::NumSpaces); ++ids) {
    DataSpace ds = static_cast<DataSpace>(ids);
    if (isDataSpaceAvailable(ds)) {
      str << getDataSpaceName(ds) << std::endl;
    }
  }
  str << "\nUnavailable data spaces:";
  str << "\n-------------------\n";
  for (int ids = 0; ids < static_cast<int>(DataSpace::NumSpaces); ++ids) {
    DataSpace ds = static_cast<DataSpace>(ids);
    if (!isDataSpaceAvailable(ds)) {
      str << getDataSpaceName(ds) << std::endl;
    }
  }
  str.flush();
}


void RunParams::printGroupNames(std::ostream& str) const
{
  str << "\nAvailable groups:";
  str << "\n-----------------\n";
  for (int gid = 0; gid < NumGroups; ++gid) {
    str << getGroupName(static_cast<GroupID>(gid)) << std::endl;
  }
  str.flush();
}

void RunParams::printFeatureNames(std::ostream& str) const
{
  str << "\nAvailable features:";
  str << "\n-------------------\n";
  for (int fid = 0; fid < NumFeatures; ++fid) {
    str << getFeatureName(static_cast<FeatureID>(fid)) << std::endl;
  }
  str.flush();
}

void RunParams::printFeatureKernels(std::ostream& str) const
{
  str << "\nAvailable features and kernels that use each:";
  str << "\n---------------------------------------------\n";
  for (int fid = 0; fid < NumFeatures; ++fid) {
    FeatureID tfid = static_cast<FeatureID>(fid);
    str << getFeatureName(tfid) << std::endl;
    for (int kid = 0; kid < NumKernels; ++kid) {
      KernelID tkid = static_cast<KernelID>(kid);
      KernelBase* kern = getKernelObject(tkid, *this);
      if ( kern->usesFeature(tfid) ) {
        str << "\t" << getFullKernelName(tkid) << std::endl;
      }
      delete kern;
    }  // loop over kernels
    str << std::endl;
  }  // loop over features
  str.flush();
}

void RunParams::printKernelFeatures(std::ostream& str) const
{
  str << "\nAvailable kernels and features each uses:";
  str << "\n-----------------------------------------\n";
  for (int kid = 0; kid < NumKernels; ++kid) {
    KernelID tkid = static_cast<KernelID>(kid);
    str << getFullKernelName(tkid) << std::endl;
    KernelBase* kern = getKernelObject(tkid, *this);
    for (int fid = 0; fid < NumFeatures; ++fid) {
      FeatureID tfid = static_cast<FeatureID>(fid);
      if ( kern->usesFeature(tfid) ) {
         str << "\t" << getFeatureName(tfid) << std::endl;
      }
    }  // loop over features
    delete kern;
  }  // loop over kernels
  str.flush();
}

}  // closing brace for rajaperf namespace
