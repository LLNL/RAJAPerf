/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Basic kernel REDUCE3_INT.
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


#ifndef RAJAPerf_Basic_REDUCE3_INT_HPP
#define RAJAPerf_Basic_REDUCE3_INT_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class REDUCE3_INT : public KernelBase
{
public:

  REDUCE3_INT(const RunParams& params);

  ~REDUCE3_INT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Int_ptr m_vec;
  Int_type m_vsum;
  Int_type m_vsum_init;
  Int_type m_vmax;
  Int_type m_vmax_init;
  Int_type m_vmin;
  Int_type m_vmin_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
