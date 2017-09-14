/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Basic kernel DIFF_PREDICT.
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
// For more information, please see the file LICENSE in the top-level directory.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_DIFF_PREDICT_HPP
#define RAJAPerf_Basic_DIFF_PREDICT_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class DIFF_PREDICT : public KernelBase
{
public:

  DIFF_PREDICT(const RunParams& params);

  ~DIFF_PREDICT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_px;
  Real_ptr m_cx;

  Index_type m_offset;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
