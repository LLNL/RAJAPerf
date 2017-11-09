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


#ifndef RAJAPerf_POLYBENCH_ADI_HXX
#define RAJAPerf_POLYBENCH_ADI_HXX

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_ADI : public KernelBase
{
public:

  POLYBENCH_ADI(const RunParams& params);

  ~POLYBENCH_ADI();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Index_type m_n;
  Index_type m_tsteps;
  Index_type m_run_reps;
  Real_ptr m_U;
  Real_ptr m_V;
  Real_ptr m_P;
  Real_ptr m_Q;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
