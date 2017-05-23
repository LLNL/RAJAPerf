/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Polybench kernel 2mm .
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


#ifndef RAJAPerf_POLYBENCH_3MM_HXX
#define RAJAPerf_POLYBENCH_3MM_HXX

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_3MM : public KernelBase
{
public:

  POLYBENCH_3MM(const RunParams& params);

  ~POLYBENCH_3MM();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nl;
  Index_type m_nm;
  Index_type m_run_samples;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
  Real_ptr m_E;
  Real_ptr m_F;
  Real_ptr m_G;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
