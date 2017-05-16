/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel ENERGY.
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


#ifndef RAJAPerf_Apps_ENERGY_HPP
#define RAJAPerf_Apps_ENERGY_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{

class ENERGY : public KernelBase
{
public:

  ENERGY(const RunParams& params);

  ~ENERGY();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_e_new;
  Real_ptr m_e_old;
  Real_ptr m_delvc;
  Real_ptr m_p_new;
  Real_ptr m_p_old; 
  Real_ptr m_q_new; 
  Real_ptr m_q_old; 
  Real_ptr m_work; 
  Real_ptr m_compHalfStep; 
  Real_ptr m_pHalfStep; 
  Real_ptr m_bvc; 
  Real_ptr m_pbvc; 
  Real_ptr m_ql_old; 
  Real_ptr m_qq_old; 
  Real_ptr m_vnewc; 

  Real_type m_rho0;
  Real_type m_e_cut;
  Real_type m_emin;
  Real_type m_q_cut;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
