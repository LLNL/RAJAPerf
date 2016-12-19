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

#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
namespace basic
{

MULADDSUB::MULADDSUB()
  : KernelBase(rajaperf::Basic_MULADDSUB)
{
}

MULADDSUB::~MULADDSUB() 
{
}

void MULADDSUB::setUp(VariantID vid)
{
}

void MULADDSUB::executeKernel(VariantID vid, const RunParams& params)
{
  RAJA::Real_ptr out1;
  RAJA::Real_ptr out2;
  RAJA::Real_ptr out3;
  RAJA::Real_ptr in1;
  RAJA::Real_ptr in2;

  RAJA::forall<RAJA::seq_exec>(0, 100, [=](int i) {
    out1[i] = in1[i] * in2[i] ;
    out2[i] = in1[i] + in2[i] ;
    out3[i] = in1[i] - in2[i] ;
  }); 
}

void MULADDSUB::computeChecksum(VariantID vid)
{
  checksum[vid] = 0.0;
}

void MULADDSUB::tearDown(VariantID vid)
{
}

} // end namespace basic
} // end namespace rajaperf
