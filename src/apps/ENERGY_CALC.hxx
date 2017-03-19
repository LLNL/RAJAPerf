/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel ENERGY_CALC.
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


#ifndef RAJAPerf_Apps_ENERGY_CALC_HXX
#define RAJAPerf_Apps_ENERGY_CALC_HXX

#include "common/KernelBase.hxx"

#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class ENERGY_CALC : public KernelBase
{
public:

  ENERGY_CALC(const RunParams& params);

  ~ENERGY_CALC();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  RAJA::Real_ptr m_e_new;
  RAJA::Real_ptr m_e_old;
  RAJA::Real_ptr m_delvc;
  RAJA::Real_ptr m_p_new;
  RAJA::Real_ptr m_p_old; 
  RAJA::Real_ptr m_q_new; 
  RAJA::Real_ptr m_q_old; 
  RAJA::Real_ptr m_work; 
  RAJA::Real_ptr m_compHalfStep; 
  RAJA::Real_ptr m_pHalfStep; 
  RAJA::Real_ptr m_bvc; 
  RAJA::Real_ptr m_pbvc; 
  RAJA::Real_ptr m_ql_old; 
  RAJA::Real_ptr m_qq_old; 
  RAJA::Real_ptr m_vnewc; 

  RAJA::Real_type m_rho0;
  RAJA::Real_type m_e_cut;
  RAJA::Real_type m_emin;
  RAJA::Real_type m_q_cut;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
