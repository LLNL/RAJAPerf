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


#ifndef RAJAPerf_Apps_COUPLE_HPP
#define RAJAPerf_Apps_COUPLE_HPP

#include "common/KernelBase.hpp"


namespace rajaperf 
{
class RunParams;

namespace apps
{
struct ADomain;

class COUPLE : public KernelBase
{
public:

  COUPLE(const RunParams& params);

  ~COUPLE();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

private:
  Complex_ptr m_t0;
  Complex_ptr m_t1;
  Complex_ptr m_t2;
  Complex_ptr m_denac;
  Complex_ptr m_denlw;

  Real_type m_clight;
  Real_type m_csound;
  Real_type m_omega0;
  Real_type m_omegar;
  Real_type m_dt;
  Real_type m_c10;
  Real_type m_fratio;
  Real_type m_r_fratio;
  Real_type m_c20;
  Complex_type m_ireal;

  Index_type m_imin;
  Index_type m_imax;
  Index_type m_jmin;
  Index_type m_jmax;
  Index_type m_kmin;
  Index_type m_kmax;

  ADomain* m_domain;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
