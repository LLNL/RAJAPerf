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


#ifndef RAJAPerf_POLYBENCH_GEMMVER_HXX
#define RAJAPerf_POLYBENCH_GEMMVER_HXX

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMMVER : public KernelBase
{
public:

  POLYBENCH_GEMMVER(const RunParams& params);

  ~POLYBENCH_GEMMVER();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Index_type m_n;
  Index_type m_run_reps;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_A;
  Real_ptr m_u1;
  Real_ptr m_v1;
  Real_ptr m_u2;
  Real_ptr m_v2;
  Real_ptr m_w;
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
