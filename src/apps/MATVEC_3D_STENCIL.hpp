//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// MATVEC_3D_STENCIL kernel reference implementation:
///
/// for (Index_type ii = ibegin; ii < iend; ++ii ) {
///   Index_type i = real_zones[ii];
///
///   b[i] = dbl[i] * xdbl[i] + dbc[i] * xdbc[i] + dbr[i] * xdbr[i] +
///          dcl[i] * xdcl[i] + dcc[i] * xdcc[i] + dcr[i] * xdcr[i] +
///          dfl[i] * xdfl[i] + dfc[i] * xdfc[i] + dfr[i] * xdfr[i] +
///
///          cbl[i] * xcbl[i] + cbc[i] * xcbc[i] + cbr[i] * xcbr[i] +
///          ccl[i] * xccl[i] + ccc[i] * xccc[i] + ccr[i] * xccr[i] +
///          cfl[i] * xcfl[i] + cfc[i] * xcfc[i] + cfr[i] * xcfr[i] +
///
///          ubl[i] * xubl[i] + ubc[i] * xubc[i] + ubr[i] * xubr[i] +
///          ucl[i] * xucl[i] + ucc[i] * xucc[i] + ucr[i] * xucr[i] +
///          ufl[i] * xufl[i] + ufc[i] * xufc[i] + ufr[i] * xufr[i] ;
///
/// }
///

#ifndef RAJAPerf_Apps_MATVEC_3D_STENCIL_HPP
#define RAJAPerf_Apps_MATVEC_3D_STENCIL_HPP

#define MATVEC_3D_STENCIL_DATA_SETUP \
  Real_ptr b = m_b; \
  \
  Real_ptr xdbl = m_x - m_domain->kp - m_domain->jp - 1 ; \
  Real_ptr xdbc = m_x - m_domain->kp - m_domain->jp     ; \
  Real_ptr xdbr = m_x - m_domain->kp - m_domain->jp + 1 ; \
  Real_ptr xdcl = m_x - m_domain->kp                - 1 ; \
  Real_ptr xdcc = m_x - m_domain->kp                    ; \
  Real_ptr xdcr = m_x - m_domain->kp                + 1 ; \
  Real_ptr xdfl = m_x - m_domain->kp + m_domain->jp - 1 ; \
  Real_ptr xdfc = m_x - m_domain->kp + m_domain->jp     ; \
  Real_ptr xdfr = m_x - m_domain->kp + m_domain->jp + 1 ; \
  Real_ptr xcbl = m_x                - m_domain->jp - 1 ; \
  Real_ptr xcbc = m_x                - m_domain->jp     ; \
  Real_ptr xcbr = m_x                - m_domain->jp + 1 ; \
  Real_ptr xccl = m_x                               - 1 ; \
  Real_ptr xccc = m_x                                   ; \
  Real_ptr xccr = m_x                               + 1 ; \
  Real_ptr xcfl = m_x                + m_domain->jp - 1 ; \
  Real_ptr xcfc = m_x                + m_domain->jp     ; \
  Real_ptr xcfr = m_x                + m_domain->jp + 1 ; \
  Real_ptr xubl = m_x + m_domain->kp - m_domain->jp - 1 ; \
  Real_ptr xubc = m_x + m_domain->kp - m_domain->jp     ; \
  Real_ptr xubr = m_x + m_domain->kp - m_domain->jp + 1 ; \
  Real_ptr xucl = m_x + m_domain->kp                - 1 ; \
  Real_ptr xucc = m_x + m_domain->kp                    ; \
  Real_ptr xucr = m_x + m_domain->kp                + 1 ; \
  Real_ptr xufl = m_x + m_domain->kp + m_domain->jp - 1 ; \
  Real_ptr xufc = m_x + m_domain->kp + m_domain->jp     ; \
  Real_ptr xufr = m_x + m_domain->kp + m_domain->jp + 1 ; \
  \
  Real_ptr dbl = m_matrix.dbl; \
  Real_ptr dbc = m_matrix.dbc; \
  Real_ptr dbr = m_matrix.dbr; \
  Real_ptr dcl = m_matrix.dcl; \
  Real_ptr dcc = m_matrix.dcc; \
  Real_ptr dcr = m_matrix.dcr; \
  Real_ptr dfl = m_matrix.dfl; \
  Real_ptr dfc = m_matrix.dfc; \
  Real_ptr dfr = m_matrix.dfr; \
  Real_ptr cbl = m_matrix.cbl; \
  Real_ptr cbc = m_matrix.cbc; \
  Real_ptr cbr = m_matrix.cbr; \
  Real_ptr ccl = m_matrix.ccl; \
  Real_ptr ccc = m_matrix.ccc; \
  Real_ptr ccr = m_matrix.ccr; \
  Real_ptr cfl = m_matrix.cfl; \
  Real_ptr cfc = m_matrix.cfc; \
  Real_ptr cfr = m_matrix.cfr; \
  Real_ptr ubl = m_matrix.ubl; \
  Real_ptr ubc = m_matrix.ubc; \
  Real_ptr ubr = m_matrix.ubr; \
  Real_ptr ucl = m_matrix.ucl; \
  Real_ptr ucc = m_matrix.ucc; \
  Real_ptr ucr = m_matrix.ucr; \
  Real_ptr ufl = m_matrix.ufl; \
  Real_ptr ufc = m_matrix.ufc; \
  Real_ptr ufr = m_matrix.ufr; \
  \
  Index_ptr real_zones = m_real_zones;

#define MATVEC_3D_STENCIL_BODY_INDEX \
  Index_type i = real_zones[ii];

#define MATVEC_3D_STENCIL_BODY \
  b[i] = dbl[i] * xdbl[i] + dbc[i] * xdbc[i] + dbr[i] * xdbr[i] + \
         dcl[i] * xdcl[i] + dcc[i] * xdcc[i] + dcr[i] * xdcr[i] + \
         dfl[i] * xdfl[i] + dfc[i] * xdfc[i] + dfr[i] * xdfr[i] + \
                                                                  \
         cbl[i] * xcbl[i] + cbc[i] * xcbc[i] + cbr[i] * xcbr[i] + \
         ccl[i] * xccl[i] + ccc[i] * xccc[i] + ccr[i] * xccr[i] + \
         cfl[i] * xcfl[i] + cfc[i] * xcfc[i] + cfr[i] * xcfr[i] + \
                                                                  \
         ubl[i] * xubl[i] + ubc[i] * xubc[i] + ubr[i] * xubr[i] + \
         ucl[i] * xucl[i] + ucc[i] * xucc[i] + ucr[i] * xucr[i] + \
         ufl[i] * xufl[i] + ufc[i] * xufc[i] + ufr[i] * xufr[i] ; \



#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{
class ADomain;

class MATVEC_3D_STENCIL : public KernelBase
{
public:

  MATVEC_3D_STENCIL(const RunParams& params);

  ~MATVEC_3D_STENCIL();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  struct Matrix
  {
    Real_ptr dbl;
    Real_ptr dbc;
    Real_ptr dbr;
    Real_ptr dcl;
    Real_ptr dcc;
    Real_ptr dcr;
    Real_ptr dfl;
    Real_ptr dfc;
    Real_ptr dfr;
    Real_ptr cbl;
    Real_ptr cbc;
    Real_ptr cbr;
    Real_ptr ccl;
    Real_ptr ccc;
    Real_ptr ccr;
    Real_ptr cfl;
    Real_ptr cfc;
    Real_ptr cfr;
    Real_ptr ubl;
    Real_ptr ubc;
    Real_ptr ubr;
    Real_ptr ucl;
    Real_ptr ucc;
    Real_ptr ucr;
    Real_ptr ufl;
    Real_ptr ufc;
    Real_ptr ufr;
  };

  Real_ptr m_b;
  Real_ptr m_x;
  Matrix m_matrix;

  ADomain* m_domain;
  Index_type* m_real_zones;
  Index_type m_zonal_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
