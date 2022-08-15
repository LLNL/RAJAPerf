//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_MAT_VEC_MULT_4GROUPS_HPP
#define RAJAPerf_Basic_MAT_VEC_MULT_4GROUPS_HPP

#define MAT_VEC_MULT_4GROUPS_DATA_SETUP \
    using vec_t = RAJA::expt::VectorRegister<Real_type>;   \
    using idx_t = RAJA::expt::VectorIndex<int, vec_t>;  \
                                                        \
    using Mat = RAJA::View<Real_type, RAJA::StaticLayout<RAJA::PERM_JI, 4,4>>; \
    using Vec = RAJA::View<Real_type, RAJA::StaticLayout<RAJA::PERM_I, 4>>;    \
    auto vall = idx_t::static_all();






#define MAT_VEC_MULT_4GROUPS_BODY           \
    Real_type * RAJA_RESTRICT  a = m_a;      \
    Real_type * RAJA_RESTRICT  x = m_x;      \
    Real_type * RAJA_RESTRICT  y = m_y;      \
    auto aa   = Mat ( &a[16 * i] );  \
    auto x0   = Vec ( &x[16 * i + 0] );  \
    auto x1   = Vec ( &x[16 * i + 4] );  \
    auto x2   = Vec ( &x[16 * i + 8] );  \
    auto x3   = Vec ( &x[16 * i +12] );  \
    auto y0   = Vec ( &y[16 * i + 0] );  \
    auto y1   = Vec ( &y[16 * i + 4] );  \
    auto y2   = Vec ( &y[16 * i + 8] );  \
    auto y3   = Vec ( &y[16 * i +12] );  \
    y0(vall) = aa(0,0) * x0(vall) + aa(0,1) * x1(vall) + aa(0,2) * x2(vall) + aa(0,3) * x3(vall); \
    y1(vall) = aa(1,0) * x0(vall) + aa(1,1) * x1(vall) + aa(1,2) * x2(vall) + aa(1,3) * x3(vall); \
    y2(vall) = aa(2,0) * x0(vall) + aa(2,1) * x1(vall) + aa(2,2) * x2(vall) + aa(2,3) * x3(vall); \
    y3(vall) = aa(3,0) * x0(vall) + aa(3,1) * x1(vall) + aa(3,2) * x2(vall) + aa(3,3) * x3(vall); 




#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class MAT_VEC_MULT_4GROUPS : public KernelBase
{
public:

  MAT_VEC_MULT_4GROUPS(const RunParams& params);

  ~MAT_VEC_MULT_4GROUPS();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void setCudaTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);


private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_a;
  Real_ptr m_x;
  Real_ptr m_y;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
