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


#ifndef RAJAPerf_Basic_INT_PREDICT_HPP
#define RAJAPerf_Basic_INT_PREDICT_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class INT_PREDICT : public KernelBase
{
public:

  INT_PREDICT(const RunParams& params);

  ~INT_PREDICT();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_type m_px_initval;
  Index_type m_offset;

  Real_ptr m_px;

  Real_type m_dm22;
  Real_type m_dm23;
  Real_type m_dm24;
  Real_type m_dm25;
  Real_type m_dm26;
  Real_type m_dm27;
  Real_type m_dm28;
  Real_type m_c0;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
