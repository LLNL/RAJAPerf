/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel MULADDSUB.
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


#ifndef RAJAPerf_Apps_PRESSURE_CALC_HXX
#define RAJAPerf_Apps_PRESSURE_CALC_HXX

#include "common/KernelBase.hxx"

#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class PRESSURE_CALC : public KernelBase
{
public:

  PRESSURE_CALC(const RunParams& params);

  ~PRESSURE_CALC();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void computeChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  RAJA::Real_ptr m_compression;
  RAJA::Real_ptr m_bvc;
  RAJA::Real_ptr m_p_new;
  RAJA::Real_ptr m_e_old;
  RAJA::Real_ptr m_vnewc; 

  RAJA::Real_type m_cls;
  RAJA::Real_type m_p_cut;
  RAJA::Real_type m_pmin;
  RAJA::Real_type m_eosvmax;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
