//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE


NESTED_INIT::NESTED_INIT(const RunParams& params)
  : KernelBase(rajaperf::Basic_NESTED_INIT, params)
{
  m_n_init = 100;

  setDefaultProblemSize(m_n_init * m_n_init * m_n_init);
  setDefaultReps(1000);

  auto n_final = std::cbrt( getTargetProblemSize() );
  m_ni = n_final;
  m_nj = n_final;
  m_nk = n_final;
  m_array_length = m_ni * m_nj * m_nk;

  setActualProblemSize( m_array_length );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 0*sizeof(Real_type)) * getActualProblemSize() );
  setFLOPsPerRep(3 * getActualProblemSize());

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

  setVariantDefined( Kokkos_Lambda );
}

NESTED_INIT::~NESTED_INIT()
{
}

void NESTED_INIT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_array, m_array_length, 0.0, vid);
}

void NESTED_INIT::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_array, m_array_length, vid);
}

void NESTED_INIT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_array, vid);
}

} // end namespace basic
} // end namespace rajaperf
