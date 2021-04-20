//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DAXPY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] += a * x[i] ;
/// }
///

#ifndef RAJAPerf_Basic_MAT_MAT_SHARED_HPP
#define RAJAPerf_Basic_MAT_MAT_SHARED_HPP

#define TL_SZ 256

#define MAT_MAT_SHARED_DATA_SETUP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C;

#define MAT_MAT_SHARED_BODY_0  \
 RAJA_TEAM_SHARED double As[TL_SZ][TL_SZ]; \
 RAJA_TEAM_SHARED double Bs[TL_SZ][TL_SZ]; \
 RAJA_TEAM_SHARED double Cs[TL_SZ][TL_SZ];


#define MAT_MAT_SHARED_BODY_1  \
  Cs[ty][tx] = 0;

#define MAT_MAT_SHARED_BODY_2  \
  const int Row = by*DEVICE_BLOCK_SIZE + ty; \
  const int Col = bx*DEVICE_BLOCK_SIZE + tx; \
  if (k*DEVICE_BLOCK_SIZE + tx < N && Row < N) \
    As[ty][tx] = A[Row*N + k*DEVICE_BLOCK_SIZE + tx]; \
  else \
   As[ty][tx] = 0.0; \
  if (k*DEVICE_BLOCK_SIZE + ty < N && Col < N)              \
    Bs[ty][tx] = Bview((k*DEVICE_BLOCK_SIZE + ty), Col); \
  else \
    Bs[ty][tx] = 0.0; \

#define MAT_MAT_SHARED_BODY_3 \
  Cs[ty][tx] += As[ty][n] * Bs[n][tx];

#define MAT_MAT_SHARED_BODY_4 \
  if(Row < N && Col < N) \
    C[Col + N*Row] = Cs[ty][tx];


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class MAT_MAT_SHARED : public KernelBase
{
public:

  MAT_MAT_SHARED(const RunParams& params);

  ~MAT_MAT_SHARED();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
