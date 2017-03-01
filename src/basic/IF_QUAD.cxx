/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel IF_QUAD.
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


#include "IF_QUAD.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define KERNEL_DATA 

#define KERNEL_BODY(i) 


IF_QUAD::IF_QUAD(const RunParams& params)
  : KernelBase(rajaperf::Basic_IF_QUAD, params)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

IF_QUAD::~IF_QUAD() 
{
}

//
// NOTE: Setup and execute methods are implemented using switch statements
//       for now. We may want to break the variants out differently...
//
void IF_QUAD::setUp(VariantID vid)
{
  switch ( vid ) {

    case Baseline : 
    case RAJA_Serial : 
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

void IF_QUAD::runKernel(VariantID vid)
{
  int run_size = getRunSize();
  int run_samples = getRunSamples();

  std::cout << "\nIF_QUAD::runKernel, vid = " << vid << std::endl;
  std::cout << "\trun_samples = " << run_samples << std::endl;
  std::cout << "\trun_size = " << run_size << std::endl;

  switch ( vid ) {

    case Baseline : {

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

    case RAJA_Serial : {

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

void IF_QUAD::computeChecksum(VariantID vid)
{
  // Overloaded methods in common to update checksum based on type
}

void IF_QUAD::tearDown(VariantID vid)
{
  switch ( vid ) {

    case Baseline :
    case RAJA_Serial :
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
