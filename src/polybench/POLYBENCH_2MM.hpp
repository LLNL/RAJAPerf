//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_2MM kernel reference implementation:
///
/// D := alpha*A*B*C + beta*D
///
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type j = 0; j < nj; j++) {
///     tmp[i][j] = 0.0;
///     for (Index_type k = 0; k < nk; ++k) {
///       tmp[i][j] += alpha * A[i][k] * B[k][j];
///     }
///   }
/// } 
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type l = 0; l < nl; l++) {
///     D[i][l] *= beta;  // NOTE: Changed to 'D[i][l] = beta;' 
///                       // to avoid need for memset operation
///                       // to zero out matrix.
///     for (Index_type j = 0; j < nj; ++j) {
///       D[i][l] += tmp[i][j] * C[j][l];
///     } 
///   }
/// } 
///


#ifndef RAJAPerf_POLYBENCH_2MM_HPP
#define RAJAPerf_POLYBENCH_2MM_HPP


#define POLYBENCH_2MM_DATA_SETUP \
  Real_ptr tmp = m_tmp; \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
  const Index_type nl = m_nl;


#define POLYBENCH_2MM_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_2MM_BODY2 \
  dot += alpha * A[k + i*nk] * B[j + k*nj];\

#define POLYBENCH_2MM_BODY3 \
  tmp[j + i*nj] = dot;

#define POLYBENCH_2MM_BODY4 \
  Real_type dot = beta;

#define POLYBENCH_2MM_BODY5 \
  dot += tmp[j + i*nj] * C[l + j*nl];

#define POLYBENCH_2MM_BODY6 \
  D[l + i*nl] = dot;


#define POLYBENCH_2MM_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_2MM_BODY2_RAJA \
  dot += alpha * Aview(i,k) * Bview(k,j);

#define POLYBENCH_2MM_BODY3_RAJA \
  tmpview(i,j) = dot;

#define POLYBENCH_2MM_BODY4_RAJA \
  dot = beta;

#define POLYBENCH_2MM_BODY5_RAJA \
  dot += tmpview(i,j) * Cview(j, l);

#define POLYBENCH_2MM_BODY6_RAJA \
  Dview(i,l) = dot;


#define POLYBENCH_2MM_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE tmpview(tmp, RAJA::Layout<2>(ni, nj)); \
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(nj, nl)); \
  VIEW_TYPE Dview(D, RAJA::Layout<2>(ni, nl));

#define POLYBENCH_2MM_DATA_VEC_SETUP \
  RAJA_INDEX_VALUE_T(II, Int_type, "II"); \
  RAJA_INDEX_VALUE_T(IJ, Int_type, "IJ"); \
  RAJA_INDEX_VALUE_T(IK, Int_type, "IK"); \
  RAJA_INDEX_VALUE_T(IL, Int_type, "IL"); \
  using matrix_t = RAJA::RegisterMatrix<Real_type, RAJA::MATRIX_ROW_MAJOR>; \
  std::array<RAJA::idx_t, 2> perm {{0,1}}; \
  RAJA::TypedView<Real_type, RAJA::Layout<2, Int_type, 1>, II, IK> Aview(A, RAJA::make_permuted_layout({{ni, nk}}, perm)); \
  RAJA::TypedView<Real_type, RAJA::Layout<2, Int_type, 1>, IK, IJ> Bview(B, RAJA::make_permuted_layout({{nk, nj}}, perm)); \
  RAJA::TypedView<Real_type, RAJA::Layout<2, Int_type, 1>, II, IJ> Tmpview(tmp, RAJA::make_permuted_layout({{ni, nj}}, perm)); \
  RAJA::TypedView<Real_type, RAJA::Layout<2, Int_type, 1>, IJ, IL> Cview(C, RAJA::make_permuted_layout({{nj, nl}}, perm)); \
  RAJA::TypedView<Real_type, RAJA::Layout<2, Int_type, 1>, II, IL> Dview(D, RAJA::make_permuted_layout({{ni, nl}}, perm)); \
  using RowA = RAJA::RowIndex<II, matrix_t>; \
  using ColA = RAJA::ColIndex<IK, matrix_t>; \
  using ColB = RAJA::ColIndex<IJ, matrix_t>; \
  using RowT = RAJA::RowIndex<II, matrix_t>; \
  using ColT = RAJA::ColIndex<IJ, matrix_t>; \
  using ColC = RAJA::ColIndex<IL, matrix_t>; \
  using EXECPOL = \
    RAJA::KernelPolicy< \
      RAJA::statement::For<2, RAJA::matrix_col_exec<matrix_t>, \
        RAJA::statement::For<1, RAJA::matrix_col_exec<matrix_t>, \
          RAJA::statement::For<0, RAJA::matrix_row_exec<matrix_t>, \
            RAJA::statement::Lambda<0> \
            > \
          > \
        > \
      >; \
  for(int i = 0; i < ni*nk; i++) { \
    A[i] = A[i] * alpha; \
  }

#define POLYBENCH_2MM_VEC_BODY1 \
  std::memset(tmp, 0, ni*nj* sizeof(Real_type));\
  for(int i = 0; i < ni*nl; i++) { \
    D[i] = beta; \
  } \
  auto segments1 = RAJA::make_tuple(RAJA::TypedRangeSegment<II>(0, ni),\
                                   RAJA::TypedRangeSegment<IK>(0, nk),\
                                   RAJA::TypedRangeSegment<IJ>(0, nj)); \
  RAJA::kernel<EXECPOL>( segments1, \
      [=] (RowA i, ColA k, ColB j) { \
          Tmpview(i, j) += Aview(i, k) * Bview(toRowIndex(k), j); \
      } \
  );\
  auto segments2 = RAJA::make_tuple(RAJA::TypedRangeSegment<II>(0, ni),\
                                   RAJA::TypedRangeSegment<IJ>(0, nj),\
                                   RAJA::TypedRangeSegment<IL>(0, nl)); \
  RAJA::kernel<EXECPOL>( segments2, \
      [=] (RowT i, ColT j, ColC l) { \
          Dview(i, l) += Tmpview(i, j) * Cview(toRowIndex(j), l); \
      } \
  ); 
//  for (II i(0); i < ni; i++ ) { \
//    for (IL l(0); l < nl; l++) { \
//      dot = 0; \
//      for (IJ j(0); j < nj; j++) { \
//        dot += Tmpview(i,j) * Cview(j,l); \
//        std::cout << "Tmp and C view: " << Tmpview(i,j) << " " << Cview(j,l) << " " << dot << std::endl; \
//      } \
//      std::cout << dot << " " << Dview(i,l) << std::endl;\
//    } \
//  }
//  for (II i(0); i < ni; i++ ) { \
//    for (IJ j(0); j < nj; j++) { \
//      dot = 0; \
//      for (IK k(0); k < nk; k++) { \
//        dot += Aview(i,k) * Bview(k,j); \
//        std::cout << "A and B view: " << Aview(i,k) << " " << Bview(k,j) << " " << dot << std::endl; \
//      } \
//      std::cout << dot << " " << Tmpview(i,j) << std::endl;\
//    } \
//  }

//#define POLYBENCH_2MM_VEC_BODY1 \
//  for (Index_type i = 0; i < ni; i++ ) { \
//    for (Index_type j = 0; j < nj; j++) { \
//      dot = 0.; \
//      for (Index_type k = 0; k < nk; k++) { \
//        dot += alpha * a.get(i,k) * b.get(k,j); \
//        std::cout << "dot: " << dot << " " << a.get(i,k) << " " << a.get(k,i) << " " << A[k+i*nk] << " " << b.get(j,k) << " " << b.get(k,j) << " " << B[j+k*nj] << std::endl; \
//      } \
//      c.set(i,j,dot); \
//    } \
//  }
        //std::cout << "dot: " << dot << " " << a.get(i,k) << " " << a.get(k,i) << " " << A[i+k*ni] << " " << b.get(j,k) << " " << b.get(k,j) << " " << B[k+j*nk] << std::endl; \
  //auto c = a * b; \
  //for (Index_type i = 0; i < ni; i++ ) { \
  //  for (Index_type j = 0; j < nj; j++) { \
  //  } \
  //}

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_2MM : public KernelBase
{
public:

  POLYBENCH_2MM(const RunParams& params);

  ~POLYBENCH_2MM();


  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nl;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_tmp;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
