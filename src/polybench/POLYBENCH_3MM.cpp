//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <algorithm>


namespace rajaperf
{
namespace polybench
{


POLYBENCH_3MM::POLYBENCH_3MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_3MM, params)
{
  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1010;
  Index_type nl_default = 1000;
  Index_type nm_default = 1200;

  setDefaultProblemSize( std::max( std::max( ni_default*nj_default,
                                             nj_default*nl_default ),
                                  ni_default*nl_default ) );
  setDefaultProblemSize( ni_default * nj_default );
  setDefaultReps(2);

  m_ni = std::sqrt( getTargetProblemSize() ) + 1;
  m_nj = m_ni;
  m_nk = nk_default;
  m_nl = m_ni;
  m_nm = nm_default;


  setActualProblemSize( std::max( std::max( m_ni*m_nj, m_nj*m_nl ),
                                  m_ni*m_nl ) );

  setItsPerRep( m_ni*m_nj + m_nj*m_nl + m_ni*m_nl );
  setKernelsPerRep(3);
  setBytesPerRep( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nk +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nk +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_nj * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nm +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nl * m_nm +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nl );
  setFLOPsPerRep(2 * m_ni*m_nj*m_nk +
                 2 * m_nj*m_nl*m_nm +
                 2 * m_ni*m_nj*m_nl );

  checksum_scale_factor = 0.000000001 *
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
}

POLYBENCH_3MM::~POLYBENCH_3MM()
{
}

void POLYBENCH_3MM::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nm, vid);
  allocAndInitData(m_D, m_nm * m_nl, vid);
  allocAndInitDataConst(m_E, m_ni * m_nj, 0.0, vid);
  allocAndInitDataConst(m_F, m_nj * m_nl, 0.0, vid);
  allocAndInitDataConst(m_G, m_ni * m_nl, 0.0, vid);
}

void POLYBENCH_3MM::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_G, m_ni * m_nl, checksum_scale_factor , vid);
}

void POLYBENCH_3MM::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_A, vid);
  deallocData(m_B, vid);
  deallocData(m_C, vid);
  deallocData(m_D, vid);
  deallocData(m_E, vid);
  deallocData(m_F, vid);
  deallocData(m_G, vid);
}

} // end namespace basic
} // end namespace rajaperf
