//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Block Diagonal Matrix-Vector Product
/// reference implementation:
///
/// for (Index_type e = 0; e < NE; ++e) {
///
///    for (Index_type c = 0; c < ndofs; ++c) {
///        Real_type dot = 0;
///        for (Index_type r = 0; r < ndofs; ++r) {
///            dot += Me(r,x,e) * X(r,e);
///    }
///    Y(c,e) = dot;
///  }
///}
///
///

#ifndef RAJAPerf_Apps_BLOCK_DIAG_MAT_VEC_HPP
#define RAJAPerf_Apps_BLOCK_DIAG_MAT_VEC_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

#define BLOCK_DIAG_MAT_VEC_DATA_INIT                                              					 \
for(int ii = 0; ii != NE; ++ii){												  	     \
  for(int i = 0; i != ndof; ++i){ (&m_X[0][0])[i+(ii*ndof)] = i; }				  					 \
  for(int i = 0; i != (ndof*ndof); ++i){ (&m_Me[0][0][0])[i+(ii*ndof*ndof)] = (ndof*ndof) - 1 - i; } \
}

#define BLOCK_DIAG_MAT_VEC_DATA_SETUP                                              \
  Real_ptr Me = m_Me;                                                              \
  Real_ptr X = m_X;                                                                \
  Real_ptr Y = m_Y;

#define BLOCK_DIAG_MAT_VEC_BODY                                             \
    double dot = 0;                                                         \
    for (int r = 0; r < ndof; ++r)                                          \
    {                                                                       \
       dot += (&Me[0][0][0])[r * ndof * 1 + c * 1 + (e*ndof*ndof)] *        \
              (&X[0][0])[r * 1 + (e*ndof)];                                 \
    }                                                                       \
    (&Y[0][0])[c * 1 + (e*ndof)] = dot;

namespace rajaperf {
class RunParams;

namespace apps {

class BLOCK_DIAG_MAT_VEC : public KernelBase {
public:
  BLOCK_DIAG_MAT_VEC(const RunParams &params);

  ~BLOCK_DIAG_MAT_VEC();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size, gpu_block_size::ExactSqrt>;

  Real_ptr m_Me;
  Real_ptr m_X;
  Real_ptr m_Y;

  Index_type m_N;
  Index_type m_N_default;
  static constexpr Index_type m_ndof = 24;
  Index_type m_NE;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
