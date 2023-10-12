//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "gtest/gtest.h"

#include "common/Executor.hpp"
#include "common/KernelBase.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <cmath>

TEST(ShortSuiteTest, Basic)
{

// Assemble command line args for basic test

  std::vector< std::string > sargv{};
  sargv.emplace_back(std::string("dummy "));  // for executable name
  sargv.emplace_back(std::string("--checkrun"));
  sargv.emplace_back(std::string("3"));
  sargv.emplace_back(std::string("--show-progress"));
  sargv.emplace_back(std::string("--disable-warmup"));

#if defined(RAJA_ENABLE_HIP) && \
     (HIP_VERSION_MAJOR < 5 || \
     (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR < 1))
  sargv.emplace_back(std::string("--exclude-kernels"));
  sargv.emplace_back(std::string("HALOEXCHANGE_FUSED"));
#endif

#if (defined(RAJA_COMPILER_CLANG) && __clang_major__ == 11)
  sargv.emplace_back(std::string("--exclude-kernels"));
  sargv.emplace_back(std::string("FIRST_MIN"));
#endif

  char *unit_test = getenv("RAJA_PERFSUITE_UNIT_TEST");
  if (unit_test != NULL) {
    sargv.emplace_back(std::string("-k"));
    sargv.emplace_back(std::string(unit_test));
  }

  char** argv = new char* [sargv.size()];
  for (size_t is = 0; is < sargv.size(); ++is) {
    argv[is] = const_cast<char*>(sargv[is].c_str());
  }

  // STEP 1: Create suite executor object with input args defined above
  rajaperf::Executor executor(sargv.size(), argv);

  // STEP 2: Assemble kernels and variants to run
  executor.setupSuite();

  // STEP 3: Report suite run summary
  executor.reportRunSummary(std::cout);

  // STEP 4: Execute suite
  executor.runSuite();

  // STEP 5: Access suite run data and run through checks
  std::vector<rajaperf::KernelBase*> kernels = executor.getKernels();
  std::vector<rajaperf::VariantID> variant_ids = executor.getVariantIDs();


  for (size_t ik = 0; ik < kernels.size(); ++ik) {

    rajaperf::KernelBase* kernel = kernels[ik];

    // 
    // Get reference checksum (first kernel variant run)
    //
    rajaperf::Checksum_type cksum_ref = 0.0;
    size_t ivck = 0;
    bool found_ref = false;
    while ( ivck < variant_ids.size() && !found_ref ) {

      rajaperf::VariantID vid = variant_ids[ivck];
      size_t num_tunings = kernel->getNumVariantTunings(vid);
      for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
        if ( kernel->wasVariantTuningRun(vid, tune_idx) ) {
          cksum_ref = kernel->getChecksum(vid, tune_idx);
          found_ref = true;
          break;
        }
      }
      ++ivck;

    } // while loop over variants until reference checksum found


    //
    // Check execution time is greater than zero and checksum diff is 
    // within tolerance for each variant run.
    // 
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {

      rajaperf::VariantID vid = variant_ids[iv];

      size_t num_tunings = kernel->getNumVariantTunings(variant_ids[iv]);
      for (size_t tune_idx = 0; tune_idx < num_tunings; ++tune_idx) {
        if ( kernel->wasVariantTuningRun(vid, tune_idx) ) {

          double rtime = kernel->getTotTime(vid, tune_idx);

          rajaperf::Checksum_type cksum = kernel->getChecksum(vid, tune_idx); 
          rajaperf::Checksum_type cksum_diff = std::abs(cksum_ref - cksum);

          // Print kernel information when running test manually
          std::cout << "Check kernel, variant, tuning : "
                    << kernel->getName() << " , "
                    << rajaperf::getVariantName(vid) << " , "
                    << kernel->getVariantTuningName(vid, tune_idx) 
                    << std::endl;
          EXPECT_GT(rtime, 0.0);
          EXPECT_LT(cksum_diff, 1e-7);
          
        }
      } 

    }  // loop over variants

  } // loop over kernels

  // clean up 
  delete [] argv; 
}
