/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel INIT3.
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


#include "INIT3.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define KERNEL_DATA 

#define KERNEL_BODY(i) 


INIT3::INIT3(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT3, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

INIT3::~INIT3() 
{
}

//
// NOTE: Setup and execute methods are implemented using switch statements
//       for now. We may want to break the variants out differently...
//
void INIT3::setUp(VariantID vid)
{
  switch ( vid ) {

    case Baseline_Seq : 
    case RAJA_Seq : 
    case Baseline_OpenMP :
    case RAJA_OpenMP : {
// Overloaded methods in common to allocate data based on array length and type
      break;
    }

    case Baseline_CUDA : 
    case RAJA_CUDA : {
      // Allocate host and device memory here.
      break;
    }

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }

  //
  // Initialize data arrays based on VariantID...
  // Use centralized methods...
  //

}

void INIT3::runKernel(VariantID vid)
{
  int run_size = getRunSize();
  int run_samples = getRunSamples();

  std::cout << "\nINIT3::runKernel, vid = " << vid << std::endl;
  std::cout << "\trun_samples = " << run_samples << std::endl;
  std::cout << "\trun_size = " << run_size << std::endl;

  switch ( vid ) {

    case Baseline_Seq : {

       KERNEL_DATA;
  
       startTimer();
       for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
//       for (RAJA::Index_type i = 0; i < run_size; ++i ) {
//         KERNEL_BODY(i);
//       }
       }
       stopTimer();

       break;
    } 

    case RAJA_Seq : {

       KERNEL_DATA;
  
       startTimer();
       for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
//       RAJA::forall<RAJA::seq_exec>(0, 100, [=](int i) {
//         KERNEL_BODY(i);
//       }); 
       }
       stopTimer(); 

       break;
    }

    case Baseline_OpenMP :
    case RAJA_OpenMP : 
    case Baseline_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
      break;
    }

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }
}

void INIT3::computeChecksum(VariantID vid)
{
  (void) vid;
  // Overloaded methods in common to update checksum based on type
}

void INIT3::tearDown(VariantID vid)
{
  switch ( vid ) {

    case Baseline_Seq :
    case RAJA_Seq :
    case Baseline_OpenMP :
    case RAJA_OpenMP : {
// Overloaded methods in common to deallocate data
      break;
    }

    case Baseline_CUDA :
    case RAJA_CUDA : {
      // De-allocate host and device memory here.
      break;
    }

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }
}

} // end namespace basic
} // end namespace rajaperf
