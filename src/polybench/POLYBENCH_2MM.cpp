//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <algorithm>


namespace rajaperf
{
namespace polybench
{

POLYBENCH_2MM::POLYBENCH_2MM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_2MM, params)
{
  Index_type ni_default = 1000;
  Index_type nj_default = 1000;
  Index_type nk_default = 1120;
  Index_type nl_default = 1000;

  setDefaultProblemSize( std::max( ni_default*nj_default,
                                   ni_default*nl_default ) );
  setDefaultReps(2);

  m_ni = std::sqrt( getTargetProblemSize() ) + 1;
  m_nj = m_ni;
  m_nk = nk_default;
  m_nl = m_ni;

  m_alpha = 1.5;
  m_beta = 1.2;


  setActualProblemSize( std::max( m_ni*m_nj, m_ni*m_nl ) );

  setItsPerRep( m_ni*m_nj + m_ni*m_nl );
  setKernelsPerRep(2);
  setBytesPerRep( (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nk +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nk +

                  (1*sizeof(Real_type ) + 0*sizeof(Real_type )) * m_ni * m_nl +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_ni * m_nj +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * m_nj * m_nl );
  setFLOPsPerRep(3 * m_ni*m_nj*m_nk +
                 2 * m_ni*m_nj*m_nl );

  checksum_scale_factor = 0.000001 *
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

POLYBENCH_2MM::~POLYBENCH_2MM()
{
}

void POLYBENCH_2MM::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  allocAndInitData(m_tmp, m_ni * m_nj, vid);
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitData(m_C, m_nj * m_nl, vid);
  allocAndInitDataConst(m_D, m_ni * m_nl, 0.0, vid);
}

void POLYBENCH_2MM::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_D, m_ni * m_nl, checksum_scale_factor );
}

void POLYBENCH_2MM::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_tmp);
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
  deallocData(m_D);
}

} // end namespace polybench
} // end namespace rajaperf
