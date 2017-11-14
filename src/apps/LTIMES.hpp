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


#ifndef RAJAPerf_Apps_LTIMES_HPP
#define RAJAPerf_Apps_LTIMES_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{

class LTIMES : public KernelBase
{
public:

  LTIMES(const RunParams& params);

  ~LTIMES();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Real_ptr m_phidat;
  Real_ptr m_elldat;
  Real_ptr m_psidat;

  Index_type m_num_d_default; 
  Index_type m_num_z_default; 
  Index_type m_num_g_default; 
  Index_type m_num_m_default; 

  Index_type m_num_d; 
  Index_type m_num_z; 
  Index_type m_num_g; 
  Index_type m_num_m; 

  Index_type m_philen;
  Index_type m_elllen;
  Index_type m_psilen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
