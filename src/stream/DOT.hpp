/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Stream kernel DOT.
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


#ifndef RAJAPerf_Stream_DOT_HPP
#define RAJAPerf_Stream_DOT_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class DOT : public KernelBase
{
public:

  DOT(const RunParams& params);

  ~DOT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_type m_dot;
  Real_type m_dot_init;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
