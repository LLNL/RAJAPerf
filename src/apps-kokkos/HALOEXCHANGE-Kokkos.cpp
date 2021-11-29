//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void HALOEXCHANGE::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();

  // Nota bene: ibegin, iend not defined for this kernel
  // Instead:
  // Index_type num_neighbors = s_num_neighbors;
  // Index_type num_vars = m_num_vars; 
  // How these variables are set::
  // apps/HALOEXCHANGE.cpp:  m_num_vars_default     = 3;
  // apps/HALOEXCHANGE.hpp:  static const int s_num_neighbors = 26;

  // HALOEXCHANGE_DATA_SETUP;

// Declare and define Kokkos Views
// Preserving the names of the pointer variables to avoid typo errors in the
// Kokkos_Lambda expressions

std::vector<Kokkos::View<Real_ptr>> vars; 
std::vector<Kokkos::View<Real_ptr>> buffers; 
std::vector<Kokkos::View<Int_ptr>>  pack_index_lists; 
std::vector<Kokkos::View<Int_ptr>>  unpack_index_lists; 

for (auto var: m_vars) {
	vars.push_back(getViewFromPointer(var, m_var_size));
}

for ( int x = 0; x < m_buffers.size(); ++x ) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[x];
	buffers.push_back(getViewFromPointer(m_buffers[x], buffer_len));
}


for ( int x = 0; x < m_pack_index_lists.size(); ++x ) {

	pack_index_lists.push_back(getViewFromPointer(m_pack_index_lists[x], m_pack_index_list_lengths[x]));
}


for ( int x = 0; x < m_unpack_index_lists.size(); ++x ) {

	unpack_index_lists.push_back(getViewFromPointer(m_unpack_index_lists[x], m_unpack_index_list_lengths[x]));
}
auto num_neighbors = s_num_neighbors;
auto num_vars = m_num_vars;

#if defined(RUN_KOKKOS)

  switch ( vid ) {

    case Kokkos_Lambda : {

	  Kokkos::fence();
      startTimer();

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        // FYI: num_neigbors defined in HALOEXCHANGE.hpp
        // num_neighbors is set in HALOEXCHANGE.cpp
        for (Index_type l = 0; l < num_neighbors; ++l) {
          auto buffer = buffers[l];
          auto list = pack_index_lists[l];
          Index_type  len  = m_pack_index_list_lengths[l];
        // FYI: num_vars defined in HALOEXCHANGE.hpp
        // num_vars is set in HALOEXCHANGE.cpp
          for (Index_type v = 0; v < num_vars; ++v) {
            auto var = vars[v];
            auto haloexchange_pack_base_lam = KOKKOS_LAMBDA(Index_type i) {
                  // HALOEXCHANGE_PACK_BODY
                  // #define HALOEXCHANGE_PACK_BODY \
                  // buffer[i] = var[list[i]];
                   buffer[i] = var[list[i]];
                };

Kokkos::parallel_for("HALOEXCHANGE - Pack Body - Kokkos Lambda",
			         Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, len),
                     haloexchange_pack_base_lam);
            //buffer += len
            
            auto end = buffer.extent(0);
            decltype(end) begin = len;
            buffer = Kokkos::subview(buffer, std::make_pair(begin,end));
          }
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          auto buffer = buffers[l];
          auto list = unpack_index_lists[l];
          Index_type  len  = m_unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            auto var = vars[v];
            auto haloexchange_unpack_base_lam = KOKKOS_LAMBDA(Index_type i) {
				//#define HALOEXCHANGE_UNPACK_BODY \
  				//var[list[i]] = buffer[i];
  				var[list[i]] = buffer[i];

                };

            Kokkos::parallel_for("HALOEXCHANGE - Unpack Body - Kokkos Lambda",
			         Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, len),
                     haloexchange_unpack_base_lam);
            //buffer += len;
            auto end = buffer.extent(0);
            decltype(end) begin = len;
            buffer = Kokkos::subview(buffer, std::make_pair(begin,end));
          }
        }

      }
      Kokkos::fence();
      stopTimer();
      break;
    }

    default : {
      std::cout << "\n HALOEXCHANGE : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

for ( int x = 0; x < m_vars.size(); ++x ) {
	//RAJAPerf Suite operation: vars.push_back(getViewFromPointer(var, m_var_size));
    moveDataToHostFromKokkosView(m_vars[x], vars[x], m_var_size);
}

for ( int x = 0; x < m_buffers.size(); ++x ) {
    Index_type buffer_len = m_num_vars * m_pack_index_list_lengths[x];
    moveDataToHostFromKokkosView(m_buffers[x], buffers[x], buffer_len);
}


for ( int x = 0; x < m_pack_index_lists.size(); ++x ) {

	//RAJAPerf Suite operation:  pack_index_lists.push_back(getViewFromPointer(m_pack_index_lists[x], m_pack_index_list_lengths[x]));
    moveDataToHostFromKokkosView(m_pack_index_lists[x], pack_index_lists[x], m_pack_index_list_lengths[x]);
}


for ( int x = 0; x < m_unpack_index_lists.size(); ++x ) {

	//unpack_index_lists.push_back(getViewFromPointer(m_unpack_index_lists[x], m_unpack_index_list_lengths[x]));
    moveDataToHostFromKokkosView(m_unpack_index_lists[x], unpack_index_lists[x], m_unpack_index_list_lengths[x]);
}

}

} // end namespace apps
} // end namespace rajaperf
