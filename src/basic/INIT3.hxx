/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Basic kernel INIT3.
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


#ifndef RAJAPerf_Basic_INIT3_HXX
#define RAJAPerf_Basic_INIT3_HXX

#include "common/KernelBase.hxx"
#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class INIT3 : public KernelBase
{
public:

  INIT3(const RunParams& params);

  ~INIT3();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void computeChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  // Data not defined yet
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
