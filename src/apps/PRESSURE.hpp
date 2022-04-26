//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

#define PRESSURE_DATA_SETUP \
  Real_ptr compression = m_compression; \
  Real_ptr bvc = m_bvc; \
  Real_ptr p_new = m_p_new; \
  Real_ptr e_old  = m_e_old; \
  Real_ptr vnewc  = m_vnewc; \
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax;


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

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

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
