//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>


namespace rajaperf
{
namespace apps
{


DEL_DOT_VEC_2D::DEL_DOT_VEC_2D(const RunParams& params)
  : KernelBase(rajaperf::Apps_DEL_DOT_VEC_2D, params)
{
  setDefaultProblemSize(1000*1000);  // See rzmax in ADomain struct
  setDefaultReps(100);

  Index_type rzmax = std::sqrt(getTargetProblemSize())+1;
  m_domain = new ADomain(rzmax, /* ndims = */ 2);

  m_array_length = m_domain->nnalls;

  setActualProblemSize(m_domain->n_real_zones);

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (0*sizeof(Index_type) + 1*sizeof(Index_type)) * getItsPerRep() +
                  (1*sizeof(Real_type)  + 0*sizeof(Real_type) ) * getItsPerRep() +
                  (0*sizeof(Real_type)  + 4*sizeof(Real_type) ) * (m_domain->imax+1-m_domain->imin)*(m_domain->jmax+1-m_domain->jmin) ) ; // touched data size, not actual number of stores and loads
  setFLOPsPerRep(54 * m_domain->n_real_zones);

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
  setVariantDefined( RAJA_StdPar );
}

DEL_DOT_VEC_2D::~DEL_DOT_VEC_2D()
{
  delete m_domain;
}

void DEL_DOT_VEC_2D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_y, m_array_length, 0.0, vid);

  Real_type dx = 0.2;
  Real_type dy = 0.1;
  setMeshPositions_2d(m_x, dx, m_y, dy, *m_domain);

  allocAndInitData(m_xdot, m_array_length, vid);
  allocAndInitData(m_ydot, m_array_length, vid);

  allocAndInitDataConst(m_div, m_array_length, 0.0, vid);

  m_ptiny = 1.0e-20;
  m_half = 0.5;
}

void DEL_DOT_VEC_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_div, m_array_length);
}

void DEL_DOT_VEC_2D::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_xdot);
  deallocData(m_ydot);
  deallocData(m_div);
}

} // end namespace apps
} // end namespace rajaperf
