//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// LTIMES kernel reference implementation:
/// Dependent on running in Kripke --layout ZGD
/// https://github.com/llnl/Kripke/blob/develop/src/Kripke/Kernel/LTimes.cpp#L27
///
/// for (Index_type z = 0; z < num_z; ++z ) {
///   for (Index_type g = 0; g < num_g; ++g ) {
///     for (Index_type m = 0; m < num_m; ++m ) {
///       for (Index_type d = 0; d < num_d; ++d ) {
///
///         phi[m+ (g * num_m) + (z * num_m * num_g)] +=
///           ell[m+ (d * num_m)] * psi[d+ (g * num_d) + (z * num_d * num_g];
///
///       }
///     }
///   }
/// }
///
/// The RAJA variants of this kernel use RAJA multi-dimensional data layouts
/// and views to do the same thing without explicit index calculations (see
/// the loop body definitions below).
///

#ifndef RAJAPerf_Apps_LTIMES_HPP
#define RAJAPerf_Apps_LTIMES_HPP

#define LTIMES_DATA_SETUP \
  ID num_d(m_num_d); \
  IZ num_z(m_num_z); \
  IG num_g(m_num_g); \
  IM num_m(m_num_m); \
\
  PSI_VIEW psi(m_psidat, \
               RAJA::make_permuted_layout( {{*num_z, *num_g, *num_d}}, \
                     RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) ); \
  ELL_VIEW ell(m_elldat, \
               RAJA::make_permuted_layout( {{*num_m, *num_d}}, \
                     RAJA::as_array<RAJA::Perm<1, 0> >::get() ) ); \
  PHI_VIEW phi(m_phidat, \
               RAJA::make_permuted_layout( {{*num_z, *num_g, *num_m}}, \
                     RAJA::as_array<RAJA::Perm<0, 1, 2> >::get() ) );

#define LTIMES_BODY \
  phi(z, g, m) +=  ell(m, d) * psi(z, g, d);


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{

//
// These index value types cannot be defined in function scope for
// RAJA CUDA variant to work.
//
namespace ltimes_idx {
  RAJA_INDEX_VALUE(ID, "ID");
  RAJA_INDEX_VALUE(IZ, "IZ");
  RAJA_INDEX_VALUE(IG, "IG");
  RAJA_INDEX_VALUE(IM, "IM");

  using PSI_VIEW = RAJA::TypedView<Real_type,
                                   RAJA::Layout<3, Index_type, 2>,
                                   IZ, IG, ID>;
  using ELL_VIEW = RAJA::TypedView<Real_type,
                                   RAJA::Layout<2, Index_type, 0>,
                                   IM, ID>;
  using PHI_VIEW = RAJA::TypedView<Real_type,
                                   RAJA::Layout<3, Index_type, 2>,
                                   IZ, IG, IM>;

  using IDRange = RAJA::TypedRangeSegment<ID>;
  using IZRange = RAJA::TypedRangeSegment<IZ>;
  using IGRange = RAJA::TypedRangeSegment<IG>;
  using IMRange = RAJA::TypedRangeSegment<IM>;
}

class LTIMES : public KernelBase
{
public:

  LTIMES(const RunParams& params);

  ~LTIMES();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  template < size_t tune_idx >
  void runSeqVariant(VariantID vid);
  template < size_t tune_idx >
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  template < size_t tune_idx >
  void runCudaVariantM(VariantID vid);
  template < size_t tune_idx, size_t block_size >
  void runCudaVariantZGM(VariantID vid);

  template < size_t tune_idx >
  void runHipVariantM(VariantID vid);
  template < size_t tune_idx, size_t block_size >
  void runHipVariantZGM(VariantID vid);

  void runSyclVariantM(VariantID vid);
  void runSyclVariantLaunchM(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantZGM(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantLaunchZGM(VariantID vid);

  template < size_t tune_idx, size_t block_size = 0 >
  void runCudaVariantImpl(VariantID vid);
  template < size_t tune_idx, size_t block_size = 0 >
  void runHipVariantImpl(VariantID vid);
  template < size_t tune_idx, size_t work_group_size = 0 >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t zgm_default_gpu_block_size = 256;
  using zgm_gpu_block_sizes_type =
      integer::make_gpu_block_size_list_type<zgm_default_gpu_block_size,
                                             integer::MultipleOf<32>>;

  Real_ptr m_phidat;
  Real_ptr m_elldat;
  Real_ptr m_psidat;

  Index_type m_num_d;
  Index_type m_num_z;
  Index_type m_num_g;
  Index_type m_num_m;

  Index_type m_philen;
  Index_type m_elllen;
  Index_type m_psilen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
