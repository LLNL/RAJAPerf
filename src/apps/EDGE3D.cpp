//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EDGE3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>


namespace rajaperf
{
namespace apps
{


EDGE3D::EDGE3D(const RunParams& params)
  : KernelBase(rajaperf::Apps_EDGE3D, params)
{
  setDefaultProblemSize(100*100*100);  // See rzmax in ADomain struct
  setDefaultReps(10);
  constexpr int num_quadrature_points = NQ_1D*NQ_1D*NQ_1D;

  Index_type rzmax = std::cbrt(getTargetProblemSize())+1;
  m_domain = new ADomain(rzmax, /* ndims = */ 3);

  m_array_length = m_domain->nnalls;
  size_t number_of_elements = m_domain->lpz+1 - m_domain->fpz;

  setActualProblemSize( number_of_elements );

  setItsPerRep( number_of_elements );
  setKernelsPerRep(1);

  constexpr size_t matrix_size = NB*NB;
  constexpr size_t basis_size = NB;

  constexpr size_t reals_per_element =
    2*NQ_1D + // quadrature size and weights
    12*basis_size + // 3 basis, 3 tbasis, 3 dbasis, 3 tdbasis
    matrix_size;

  // touched data size, not actual number of stores and loads ?
  setBytesPerRep( number_of_elements*reals_per_element*sizeof(Real_type) );

  // Only consider the operations in the innermost loop
  // these are done for each element of a matrix of size matrix_size
  // and this matrix is computed num_quadrature_points times
  constexpr size_t innermost_flops =
    6 + // detjwgts*(txm*txp + tym*typ + tzm*tzp)
    6 + // detjwgts*(dtxm*dtxp + dtym*dtyp + dtzm*dtzp)
    1;  // x = Mtemp + Stemp

  constexpr size_t flops_per_element = num_quadrature_points*matrix_size*innermost_flops;

  setFLOPsPerRep(number_of_elements * flops_per_element);

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );
  setVariantDefined( Lambda_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );
  setVariantDefined( Lambda_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

EDGE3D::~EDGE3D()
{
  delete m_domain;
}

void EDGE3D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_x, m_array_length, Real_type(0.0), vid);
  allocAndInitDataConst(m_y, m_array_length, Real_type(0.0), vid);
  allocAndInitDataConst(m_z, m_array_length, Real_type(0.0), vid);

  Real_type dx = 0.3;
  Real_type dy = 0.2;
  Real_type dz = 0.1;
  setMeshPositions_3d(m_x, dx, m_y, dy, m_z, dz, *m_domain);

  allocAndInitDataConst(m_sum, m_array_length, Real_type(0.0), vid);
}

void EDGE3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_sum, m_array_length, checksum_scale_factor );
}

void EDGE3D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(m_x, vid);
  deallocData(m_y, vid);
  deallocData(m_z, vid);

  deallocData(m_sum, vid);
}

} // end namespace apps
} // end namespace rajaperf
