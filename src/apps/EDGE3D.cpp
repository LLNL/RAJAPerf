//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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
  Index_type rzmax = std::cbrt(getTargetProblemSize())+1;
  m_domain = new ADomain(rzmax, /* ndims = */ 3);

  m_array_length = m_domain->nnalls;
  size_t number_of_elements = m_domain->lpz+1 - m_domain->fpz;

  setActualProblemSize( number_of_elements );

  setItsPerRep( number_of_elements );
  setKernelsPerRep(1);

  // touched data size, not actual number of stores and loads
  // see VOL3D.cpp
  size_t reads_per_node = 3*sizeof(Real_type);
  size_t writes_per_zone = 1*sizeof(Real_type);
  setBytesPerRep( writes_per_zone * getItsPerRep() +
                  reads_per_node * (getItsPerRep() + 1+m_domain->jp+m_domain->kp) );

  constexpr size_t flops_k_loop = 15
                                  + 6*flops_Jxx()
                                  + flops_jacobian_inv()
                                  + flops_transform_basis(EB) // flops for transform_edge_basis()
                                  + flops_transform_basis(EB) + 9 // flops for transform_curl_edge_basis()
                                  + 2*flops_inner_product<12, 12>(true);

  constexpr size_t flops_j_loop = flops_k_loop*NQ_1D + 3*flops_Jxx() + 6;
  constexpr size_t flops_i_loop = flops_j_loop*NQ_1D + 1;

  constexpr size_t flops_per_element = flops_i_loop*NQ_1D + 9*flops_Jxx() + flops_compute_detj();

  setFLOPsPerRep(number_of_elements * flops_per_element);

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

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

  {
    auto reset_x = scopedMoveData(m_x, m_array_length, vid);
    auto reset_y = scopedMoveData(m_y, m_array_length, vid);
    auto reset_z = scopedMoveData(m_z, m_array_length, vid);

    Real_type dx = 0.3;
    Real_type dy = 0.2;
    Real_type dz = 0.1;
    setMeshPositions_3d(m_x, dx, m_y, dy, m_z, dz, *m_domain);
  }

  allocAndInitDataConst(m_sum, m_array_length, Real_type(0.0), vid);
}

void EDGE3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_sum, m_array_length, checksum_scale_factor, vid  );
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
