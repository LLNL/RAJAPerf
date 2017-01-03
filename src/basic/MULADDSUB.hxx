/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Basic kernel MULADDSUB.
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


#ifndef RAJAPerf_Basic_MULADDSUB_HXX
#define RAJAPerf_Basic_MULADDSUB_HXX

#include "common/KernelBase.hxx"
#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
namespace basic
{

class MULADDSUB : public KernelBase
{
public:

  MULADDSUB(double sample_frac, double size_frac); 

  ~MULADDSUB();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void computeChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  RAJA::Real_ptr m_out1;
  RAJA::Real_ptr m_out2;
  RAJA::Real_ptr m_out3;
  RAJA::Real_ptr m_in1;
  RAJA::Real_ptr m_in2; 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
