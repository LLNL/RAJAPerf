//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_NESTED_INIT_HPP
#define RAJAPerf_Basic_NESTED_INIT_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class NESTED_INIT : public KernelBase
{
public:

  NESTED_INIT(const RunParams& params);

  ~NESTED_INIT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_array;
  Int_type m_ni;
  Int_type m_nj;
  Int_type m_nk;
  Int_type m_nk_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
