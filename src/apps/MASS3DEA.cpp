//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{


MASS3DEA::MASS3DEA(const RunParams& params)
  : KernelBase(rajaperf::Apps_MASS3DEA, params)
{
  m_NE_default = 8000;

  setDefaultProblemSize(m_NE_default*MEA_Q1D*MEA_Q1D*MEA_Q1D);
  setDefaultReps(1);

  const int ea_mat_entries = MEA_D1D*MEA_D1D*MEA_D1D*MEA_D1D*MEA_D1D*MEA_D1D;
  
  m_NE = std::max(getTargetProblemSize()/(ea_mat_entries), Index_type(1));

  setActualProblemSize( m_NE*ea_mat_entries);

  setItsPerRep(getActualProblemSize());
  setKernelsPerRep(1);

  setBytesPerRep( MEA_Q1D*MEA_D1D*sizeof(Real_type)  + // B
                  MEA_Q1D*MEA_Q1D*MEA_Q1D*m_NE*sizeof(Real_type) + // D
                  ea_mat_entries*m_NE*sizeof(Real_type) ); // M_e

  setFLOPsPerRep(m_NE * 7 * ea_mat_entries);
                 
  setUsesFeature(Launch);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

}

MASS3DEA::~MASS3DEA()
{
}

void MASS3DEA::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{

  allocAndInitDataConst(m_B, int(MEA_Q1D*MEA_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(MEA_Q1D*MEA_Q1D*MEA_Q1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_M, int(MEA_D1D*MEA_D1D*MEA_D1D*
                                 MEA_D1D*MEA_D1D*MEA_D1D*m_NE), Real_type(0.0), vid);
}

void MASS3DEA::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_M, MEA_D1D*MEA_D1D*MEA_D1D*
                                          MEA_D1D*MEA_D1D*MEA_D1D*m_NE, vid);
}

void MASS3DEA::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_B, vid);
  deallocData(m_D, vid);
  deallocData(m_M, vid);
}

} // end namespace apps
} // end namespace rajaperf
