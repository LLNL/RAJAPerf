/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Stream kernel MUL.
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


#ifndef RAJAPerf_Stream_MUL_HPP
#define RAJAPerf_Stream_MUL_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class MUL : public KernelBase
{
public:

  MUL(const RunParams& params);

  ~MUL();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
