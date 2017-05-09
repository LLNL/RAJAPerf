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


#ifndef RAJAPerf_POLYBENCH_2MM_HXX
#define RAJAPerf_POLYBENCH_2MM_HXX

#include "common/KernelBase.hxx"

namespace rajaperf 
{
class RunParams;

namespace polybench
{

class POLYBENCH_2MM : public KernelBase
{
public:

  POLYBENCH_2MM(const RunParams& params);

  ~POLYBENCH_2MM();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  // Data not defined yet
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
