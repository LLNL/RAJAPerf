/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel FIR.
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


#ifndef RAJAPerf_Apps_FIR_HPP
#define RAJAPerf_Apps_FIR_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{

class FIR : public KernelBase
{
public:

  FIR(const RunParams& params);

  ~FIR();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_in;
  Real_ptr m_out;

  Index_type m_coefflen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
