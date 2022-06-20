//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Compute D = A x B + C, where
// Inputs:
// A: N/(Ne*Ne) Ne x Ne matrices
// B: N/(Ne*Ne) Ne x Ne matrices
// Ouput:
// D: N/(Ne*Ne) Ne x Ne matrices
// All square row-major matrices, C is ignored.
//for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){
//      for(Index_type row = 0; row != Ne; ++row){
//        for(Index_type col = 0; col != Ne; ++col){
//            MAT_FUSED_MUL_ADD_BODY;
//        }   
//    }   
//} 
#ifndef RAJAPerf_Basic_MAT_FUSED_MUL_ADD_HPP
#define RAJAPerf_Basic_MAT_FUSED_MUL_ADD_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

#define MAT_FUSED_MUL_ADD_DATA_INIT                                                  \
for(Index_type ii = 0; ii != (N/(Ne*Ne)); ++ii){                                     \
	for(Index_type i = 0; i != NeNe; ++i){ m_A[i+(ii*NeNe)] = i; }               \
	for(Index_type i = 0; i != NeNe; ++i){ m_B[i+(ii*NeNe)] = NeNe - 1 - i; }    \
} 

#define MAT_FUSED_MUL_ADD_DATA_SETUP                                                 \
  Real_ptr A = m_A; 						                     \
  Real_ptr B = m_B; 								     \
  Real_ptr D = m_D; 						

#define MAT_FUSED_MUL_ADD_BODY                                               	     \
  Real_type dot = 0;                                                         	     \
  for (Index_type k = 0; k < Ne; ++k) {                                      	     \
    dot += A[row*Ne + k + ii*(Ne*Ne)] * B[k*Ne + col + ii*(Ne*Ne)];          	     \
  }                                                                          	     \
  D[row*Ne + col + ii*(Ne*Ne)] = dot;                                        	     \

namespace rajaperf {
class RunParams;

namespace basic {

class MAT_FUSED_MUL_ADD : public KernelBase {
public:
  MAT_FUSED_MUL_ADD(const RunParams &params);

  ~MAT_FUSED_MUL_ADD();

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
  void runHipVariantBuiltin(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size, gpu_block_size::ExactSqrt>;

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_D;

  Index_type m_N;
  Index_type m_N_default;
  static constexpr Index_type m_Ne = 16;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
