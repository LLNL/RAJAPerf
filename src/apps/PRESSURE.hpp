//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
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

///
/// PRESSURE kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   bvc[i] = cls * (compression[i] + 1.0);
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   p_new[i] = bvc[i] * e_old[i] ;
///   if ( fabs(p_new[i]) <  p_cut ) p_new[i] = 0.0 ;
///   if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ;
///   if ( p_new[i]  <  pmin ) p_new[i]   = pmin ;
/// }
///

#ifndef RAJAPerf_Apps_PRESSURE_HPP
#define RAJAPerf_Apps_PRESSURE_HPP


#define PRESSURE_BODY1 \
  bvc[i] = cls * (compression[i] + 1.0);

#define PRESSURE_BODY2 \
  p_new[i] = bvc[i] * e_old[i] ; \
  if ( fabs(p_new[i]) <  p_cut ) p_new[i] = 0.0 ; \
  if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ; \
  if ( p_new[i]  <  pmin ) p_new[i]   = pmin ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class PRESSURE : public KernelBase
{
public:

  PRESSURE(const RunParams& params);

  ~PRESSURE();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_compression;
  Real_ptr m_bvc;
  Real_ptr m_p_new;
  Real_ptr m_e_old;
  Real_ptr m_vnewc; 

  Real_type m_cls;
  Real_type m_p_cut;
  Real_type m_pmin;
  Real_type m_eosvmax;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
