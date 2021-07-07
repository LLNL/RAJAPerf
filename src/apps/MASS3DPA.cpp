//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


MASS3DPA::MASS3DPA(const RunParams& params)
  : KernelBase(rajaperf::Apps_MASS3DPA, params)
{
  m_NE_default = 924385;

  setDefaultSize(m_NE_default);
  setDefaultReps(50);

  setProblemSize( getRunSize() * Q1D * Q1D);

  m_NE = (getProblemSize())/(Q1D*Q1D);

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);

  setBytesPerRep( m_Q1D*m_D1D*sizeof(Real_type)  +
                  m_Q1D*m_D1D*sizeof(Real_type)  +
                  m_Q1D*m_Q1D*m_Q1D*m_NE*sizeof(Real_type) +
                  m_D1D*m_D1D*m_D1D*m_NE*sizeof(Real_type) +
                  m_D1D*m_D1D*m_D1D*m_NE*sizeof(Real_type) );

  setFLOPsPerRep(m_NE * (2 * D1D * D1D * D1D * Q1D +
                         2 * D1D * D1D * Q1D * Q1D +
                         2 * D1D * Q1D * Q1D * Q1D + Q1D * Q1D * Q1D +
                         2 * Q1D * Q1D * Q1D * D1D +
                         2 * Q1D * Q1D * D1D * D1D +
                         2 * Q1D * D1D * D1D * D1D + D1D * D1D * D1D));
  setUsesFeature(Teams);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

}

MASS3DPA::~MASS3DPA()
{
}

void MASS3DPA::setUp(VariantID vid)
{

  allocAndInitDataConst(m_B, int(m_Q1D*m_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_Bt,int(m_Q1D*m_D1D), Real_type(1.0), vid);
  allocAndInitDataConst(m_D, int(m_Q1D*m_Q1D*m_Q1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_X, int(m_D1D*m_D1D*m_D1D*m_NE), Real_type(1.0), vid);
  allocAndInitDataConst(m_Y, int(m_D1D*m_D1D*m_D1D*m_NE), Real_type(0.0), vid);
}

void MASS3DPA::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_Y, m_D1D*m_D1D*m_D1D*m_NE);
}

void MASS3DPA::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_B);
  deallocData(m_Bt);
  deallocData(m_D);
  deallocData(m_X);
  deallocData(m_Y);
}

} // end namespace apps
} // end namespace rajaperf
