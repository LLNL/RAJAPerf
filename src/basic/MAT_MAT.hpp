//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Matrix matrix multiplication with tiling, but WITHOUT shared memory
/// reference implementation:
///
/// This kernel is the companion of MAT_MAT_SHARED. It uses the identical
/// tiled decomposition of the iteration space and computes the identical
/// result, but each thread accumulates into a private scalar and reads A and
/// B directly from global memory, rather than the team cooperatively staging
/// tiles of A and B into shared memory. Because nothing is shared between
/// threads, the separate tile-load phase and the team synchronizations it
/// requires are both gone. Comparing the two kernels therefore isolates the
/// cost/benefit of the shared memory staging alone.
///
///      for (Index_type by = 0; by < Ny; ++by) {
///        for (Index_type bx = 0; bx < Nx; ++bx) {
///
///          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
///            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
///
///              const Index_type Row = by * TL_SZ + ty;
///              const Index_type Col = bx * TL_SZ + tx;
///              double dot = 0.0;
///
///              for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {
///                for (Index_type n = 0; n < TL_SZ; ++n) {
///                  const Index_type kn = k * TL_SZ + n;
///                  const double a = (kn < N && Row < N) ? A[Row * N + kn] : 0.0;
///                  const double b = (kn < N && Col < N) ? B[kn * N + Col] : 0.0;
///                  dot += a * b;
///                }
///              }
///
///              if (Row < N && Col < N)
///                C[Col + N * Row] = dot;
///            }
///          }
///        }
///      }
///
///

#ifndef RAJAPerf_Basic_MAT_MAT_HPP
#define RAJAPerf_Basic_MAT_MAT_HPP

#include "RAJA/RAJA.hpp"
#include "common/KernelBase.hpp"

//
// Tile size. Named distinctly from MAT_MAT_SHARED's TL_SZ because both
// headers are included by common/RAJAPerfSuite.cpp.
//
constexpr rajaperf::Index_type MAT_MAT_TL_SZ = 16;

#define MAT_MAT_DATA_SETUP                                                     \
  Real_ptr A = m_A;                                                            \
  Real_ptr B = m_B;                                                            \
  Real_ptr C = m_C;

//
// Unlike MAT_MAT_SHARED, no RAJA_TEAM_SHARED arrays are declared, so there is
// no BODY_0 and no need for the CLANG/HIP host-compile workaround.
//
#define MAT_MAT_BODY_1(tile_size)                                              \
  const Index_type Row = by * tile_size + ty;                                  \
  const Index_type Col = bx * tile_size + tx;                                  \
  Real_type dot = 0.0;

#define MAT_MAT_BODY_2(tile_size)                                              \
  for (Index_type n = 0; n < tile_size; ++n) {                                 \
    const Index_type kn = k * tile_size + n;                                   \
    const Real_type a_kn = (kn < N && Row < N) ? A[Row * N + kn] : 0.0;        \
    const Real_type b_kn = (kn < N && Col < N) ? B[kn * N + Col] : 0.0;        \
    dot += a_kn * b_kn;                                                        \
  }

#define MAT_MAT_BODY_3(tile_size)                                              \
  if (Row < N && Col < N)                                                      \
    C[Col + N * Row] = dot;

namespace rajaperf {
class RunParams;

namespace basic {

class MAT_MAT : public KernelBase {
public:
  MAT_MAT(const RunParams &params);

  ~MAT_MAT();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = MAT_MAT_TL_SZ * MAT_MAT_TL_SZ;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size, integer::ExactSqrt>;

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;

  Index_type m_N;
  Index_type m_N_default;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
