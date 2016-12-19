/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for LCALS kernel MULADDSUB.
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


#ifndef RAJAPerf_LCALS_MULADDSUB_HXX
#define RAJAPerf_LCALS_MULADDSUB_HXX

#include "common/KernelBase.hxx"

namespace rajaperf 
{
namespace basic
{

class MULADDSUB : public KernelBase
{
public:

  MULADDSUB(); 

  ~MULADDSUB();

  void setUp(VariantID vid);
  void executeKernel(VariantID vid, const RunParams& params); 
  void computeChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:

};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
