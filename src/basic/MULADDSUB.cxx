/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for Basic kernel MULADDSUB.
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


#include "MULADDSUB.hxx"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define KERNEL_DATA \
  RAJA::Real_ptr out1 = m_out1; \
  RAJA::Real_ptr out2 = m_out2; \
  RAJA::Real_ptr out3 = m_out3; \
  RAJA::Real_ptr in1  = m_in1; \
  RAJA::Real_ptr in2  = m_in2;

#define KERNEL_BODY(i) \
  out1[i] = in1[i] * in2[i] ; \
  out2[i] = in1[i] + in2[i] ; \
  out3[i] = in1[i] - in2[i] ;


MULADDSUB::MULADDSUB(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULADDSUB, params),
    m_out1(0),
    m_out2(0),
    m_out3(0),
    m_in1(0),
    m_in2(0)
{
   setDefaultSize(100000);
   setDefaultSamples(10000);
}

MULADDSUB::~MULADDSUB() 
{
}

//
// NOTE: Setup and execute methods are implemented using switch statements
//       for now. We may want to break the variants out differently...
//
void MULADDSUB::setUp(VariantID vid)
{
  switch ( vid ) {

    case Baseline : 
    case RAJA_Serial : 
    case Baseline_OpenMP :
    case RAJA_OpenMP : {
// Overloaded methods in common to allocate data based on array length and type
//    allocate1DAligned(m_out1, run_size);
//    allocate1DAligned(m_out2, run_size);
//    allocate1DAligned(m_out3, run_size);
//    allocate1DAligned(m_in1, run_size);
//    allocate1DAligned(m_in2, run_size);
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

    // No default. We shouldn't need one...
  }

  //
  // Initialize data arrays based on VariantID...
  // Use centralized methods...
  //

}

void MULADDSUB::runKernel(VariantID vid)
{
  std::cout << "\nMULADDSUB::runKernel, vid = " << vid << std::endl;
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

void MULADDSUB::computeChecksum(VariantID vid)
{
  // Overloaded methods in common to update checksum based on type
  //updateChksum(checksum[vid], run_size, m_out1);
  //updateChksum(checksum[vid], run_size, m_out2);
  //updateChksum(checksum[vid], run_size, m_out3);
}

void MULADDSUB::tearDown(VariantID vid)
{
  switch ( vid ) {

    case Baseline :
    case RAJA_Serial :
    case Baseline_OpenMP :
    case RAJA_OpenMP : {
// Overloaded methods in common to deallocate data
//    dallocate(m_out1, run_size);
//    dallocate(m_out2, run_size);
//    dallocate(m_out3, run_size);
//    dallocate(m_in1, run_size);
//    dallocate(m_in2, run_size);
      break;
    }

    case Baseline_CUDA :
    case RAJA_CUDA : {
      // De-allocate host and device memory here.
      break;
    }

    // No default. We shouldn't need one...
  }
}

} // end namespace basic
} // end namespace rajaperf
