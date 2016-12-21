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
#include "RAJA/RAJA.hxx"

namespace rajaperf 
{

//
// IMPORTANT -- THESE ARE TEMPORARY !!!!!
//
// We actualy want a centralized type mechanism for scalar data and 
// pointers to arrays of scalars. We can use what's in RAJA or define
// our own typedefs for the suite based on that, etc. ...TBD!! 
//
const int DATA_ALIGN = 32;
typedef RAJA::Index_type Index_type;
typedef volatile int SampIndex_type;

typedef RAJA::Real_type Real_type;
typedef Real_type* Real_ptr;


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
  Real_ptr m_out1;
  Real_ptr m_out2;
  Real_ptr m_out3;
  Real_ptr m_in1;
  Real_ptr m_in2; 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
