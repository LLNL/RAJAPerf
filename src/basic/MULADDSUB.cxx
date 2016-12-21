/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for LCALS kernel MULADDSUB.
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

#include<cstdlib>

namespace rajaperf 
{
namespace basic
{

#define KERNEL_DATA \
  Real_ptr out1 = m_out1; \
  Real_ptr out2 = m_out2; \
  Real_ptr out3 = m_out3; \
  Real_ptr in1  = m_in1; \
  Real_ptr in2  = m_in2;

#define KERNEL_BODY(i) \
  out1[i] = in1[i] * in2[i] ; \
  out2[i] = in1[i] + in2[i] ; \
  out3[i] = in1[i] - in2[i] ;

MULADDSUB::MULADDSUB(double sample_frac, double size_frac)
  : KernelBase(rajaperf::basic_MULADDSUB),
    m_out1(0),
    m_out2(0),
    m_out3(0),
    m_in1(0),
    m_in2(0)
{
   default_size    = 100000;  
   default_samples = 8000000;
   run_size        = static_cast<Index_type>(size_frac * default_size);
   run_samples     = static_cast<SampIndex_type>(sample_frac * default_samples);
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

    case BASELINE : 
    case RAJA_SERIAL : 
    case BASELINE_OPENMP :
    case RAJA_OPENMP : {
// Overloaded methods in common to allocate data based on array length and type
//    allocate1DAligned(m_out1, run_size);
//    allocate1DAligned(m_out2, run_size);
//    allocate1DAligned(m_out3, run_size);
//    allocate1DAligned(m_in1, run_size);
//    allocate1DAligned(m_in2, run_size);
      break;
    }

    case BASELINE_CUDA : 
    case RAJA_CUDA : {
      // Allocate host and device memory here.
      break;
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
  switch ( vid ) {

    case BASELINE : {

       KERNEL_DATA;
  
       startTimer();
       for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
         for (Index_type i=0 ; i < run_size ; i++ ) {
           KERNEL_BODY(i);
         }
       }
       stopTimer();

       break;
    } 

    case RAJA_SERIAL : {

       KERNEL_DATA;

       startTimer();
       for (SampIndex_type isamp = 0; isamp < run_samples; ++isamp) {
         RAJA::forall<RAJA::seq_exec>(0, 100, [=](int i) {
           KERNEL_BODY(i);
         }); 
       }
       stopTimer(); 

       break;
    }

    case BASELINE_OPENMP :
    case RAJA_OPENMP : 
    case BASELINE_CUDA :
    case RAJA_CUDA : {
      // Fill these in later...you get the idea...
      break;
    }

    // No default. We shouldn't need one...
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

    case BASELINE :
    case RAJA_SERIAL :
    case BASELINE_OPENMP :
    case RAJA_OPENMP : {
      free( m_out1 );
      free( m_out2 );
      free( m_out3 );
      free( m_in1 );
      free( m_in2 );
      break;
    }

    case BASELINE_CUDA :
    case RAJA_CUDA : {
      // De-allocate host and device memory here.
      break;
    }

    // No default. We shouldn't need one...
  }
}

} // end namespace basic
} // end namespace rajaperf
