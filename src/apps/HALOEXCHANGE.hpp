//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HALOEXCHANGE kernel reference implementation:
///
/// for (Index_type z = 0; z < num_z; ++z ) {
///   for (Index_type g = 0; g < num_g; ++g ) {
///     for (Index_type m = 0; z < num_m; ++m ) {
///       for (Index_type d = 0; d < num_d; ++d ) {
///
///         phi[m+ (g * num_m) + (z * num_m * num_g)] +=
///           ell[d+ (m * num_d)] * psi[d+ (g * num_d) + (z * num_d * num_g];
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

#ifndef RAJAPerf_Apps_HALOEXCHANGE_HPP
#define RAJAPerf_Apps_HALOEXCHANGE_HPP

#define HALOEXCHANGE_DATA_SETUP \
  std::vector<Real_ptr> vars = m_vars; \
  std::vector<Real_ptr> buffers = m_buffers; \
\
  Index_type num_neighbors = s_num_neighbors; \
  Index_type num_vars = m_num_vars; \
  std::vector<Int_ptr> pack_index_lists = m_pack_index_lists; \
  std::vector<Index_type> pack_index_list_lengths = m_pack_index_list_lengths; \
  std::vector<Int_ptr> unpack_index_lists = m_unpack_index_lists; \
  std::vector<Index_type> unpack_index_list_lengths = m_unpack_index_list_lengths;

#define HALOEXCHANGE_PACK_BODY \
  buffer[i] = var[list[i]];

#define HALOEXCHANGE_UNPACK_BODY \
  var[list[i]] = buffer[i];


#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include <vector>

namespace rajaperf
{
class RunParams;

namespace apps
{

class HALOEXCHANGE : public KernelBase
{
public:

  HALOEXCHANGE(const RunParams& params);

  ~HALOEXCHANGE();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  static const int s_num_neighbors = 26;

  Index_type m_grid_dims[3];
  Index_type m_halo_width;
  Index_type m_num_vars;

  Index_type m_grid_dims_default[3];
  Index_type m_halo_width_default;
  Index_type m_num_vars_default;

  Index_type m_grid_plus_halo_dims[3];
  Index_type m_var_size;

  std::vector<Real_ptr> m_vars;
  std::vector<Real_ptr> m_buffers;

  std::vector<Int_ptr> m_pack_index_lists;
  std::vector<Index_type > m_pack_index_list_lengths;
  std::vector<Int_ptr> m_unpack_index_lists;
  std::vector<Index_type > m_unpack_index_list_lengths;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
