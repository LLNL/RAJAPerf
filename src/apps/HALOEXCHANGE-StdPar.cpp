//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void HALOEXCHANGE::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_DATA_SETUP;

  auto begin = counting_iterator<Index_type>(0);
  auto end   = counting_iterator<Index_type>(num_neighbors);

  switch ( vid ) {

    case Base_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type  len  = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            for (Index_type i = 0; i < len; i++) {
              HALOEXCHANGE_PACK_BODY;
            }
            buffer += len;
          }
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type  len  = unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            for (Index_type i = 0; i < len; i++) {
              HALOEXCHANGE_UNPACK_BODY;
            }
            buffer += len;
          }
        });

      }
      stopTimer();

      break;
    }

    case Lambda_StdPar : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type  len  = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto haloexchange_pack_base_lam = [=](Index_type i) {
                  HALOEXCHANGE_PACK_BODY;
                };
            for (Index_type i = 0; i < len; i++) {
              haloexchange_pack_base_lam(i);
            }
            buffer += len;
          }
        });

        std::for_each( std::execution::par_unseq,
                        begin, end,
                        [=](Index_type l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type  len  = unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            auto haloexchange_unpack_base_lam = [=](Index_type i) {
                  HALOEXCHANGE_UNPACK_BODY;
                };
            for (Index_type i = 0; i < len; i++) {
              haloexchange_unpack_base_lam(i);
            }
            buffer += len;
          }
        });

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n HALOEXCHANGE : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

} // end namespace apps
} // end namespace rajaperf
