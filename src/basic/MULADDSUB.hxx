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

namespace rajaperf 
{
class RunParams;

namespace basic
{

class MULADDSUB : public KernelBase
{
public:

  MULADDSUB(const RunParams& params);

  ~MULADDSUB();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_out1;
  Real_ptr m_out2;
  Real_ptr m_out3;
  Real_ptr m_in1;
  Real_ptr m_in2; 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
