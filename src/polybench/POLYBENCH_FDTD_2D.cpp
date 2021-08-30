//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <algorithm>
#include <iostream>
#include <cstring>

namespace rajaperf
{
namespace polybench
{


POLYBENCH_FDTD_2D::POLYBENCH_FDTD_2D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FDTD_2D, params)
{
  Index_type nx_default = 1000;
  Index_type ny_default = 1000;

  setDefaultProblemSize( std::max( (nx_default-1) * ny_default, 
                                    nx_default * (ny_default-1) ) );
  setDefaultReps(8);

  m_nx = std::sqrt( getTargetProblemSize() ) + 1;
  m_ny = m_nx;
  m_tsteps = 40;


  setActualProblemSize( std::max( (m_nx-1)*m_ny, m_nx*(m_ny-1) ) ); 

  setItsPerRep( m_tsteps * ( m_ny +
                             (m_nx-1)*m_ny +
                             m_nx*(m_ny-1) +
                             (m_nx-1)*(m_ny-1) ) );
  setKernelsPerRep(m_tsteps * 4);
  setBytesPerRep( m_tsteps * ( (0*sizeof(Real_type ) + 1*sizeof(Real_type )) +
                               (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ny +

                               (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * (m_nx-1) * m_ny +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nx * m_ny +

                               (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nx * (m_ny-1) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nx * m_ny +

                               (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * (m_nx-1) * (m_ny-1) +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * (m_nx-1) * m_ny +
                               (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nx * (m_ny-1) ) );
  setFLOPsPerRep( m_tsteps * ( 0 * m_ny +
                               3 * (m_nx-1)*m_ny +
                               3 * m_nx*(m_ny-1) +
                               5 * (m_nx-1)*(m_ny-1) ) );

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Kernel);

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

POLYBENCH_FDTD_2D::~POLYBENCH_FDTD_2D()
{
}

void POLYBENCH_FDTD_2D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_hz, m_nx * m_ny, 0.0, vid);
  allocAndInitData(m_ex, m_nx * m_ny, vid);
  allocAndInitData(m_ey, m_nx * m_ny, vid);
  allocAndInitData(m_fict, m_tsteps, vid);
}

void POLYBENCH_FDTD_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_hz, m_nx * m_ny, checksum_scale_factor);
}

void POLYBENCH_FDTD_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_fict);
  deallocData(m_ex);
  deallocData(m_ey);
  deallocData(m_hz);
}

} // end namespace polybench
} // end namespace rajaperf
