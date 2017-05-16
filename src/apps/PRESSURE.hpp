/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel PRESSURE.
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


#ifndef RAJAPerf_Apps_PRESSURE_HPP
#define RAJAPerf_Apps_PRESSURE_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{

class PRESSURE : public KernelBase
{
public:

  PRESSURE(const RunParams& params);

  ~PRESSURE();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_compression;
  Real_ptr m_bvc;
  Real_ptr m_p_new;
  Real_ptr m_e_old;
  Real_ptr m_vnewc; 

  Real_type m_cls;
  Real_type m_p_cut;
  Real_type m_pmin;
  Real_type m_eosvmax;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
