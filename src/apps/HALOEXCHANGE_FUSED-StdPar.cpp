//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALOEXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void HALOEXCHANGE_FUSED::runStdParVariant(VariantID vid, size_t tune_idx)
{
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  HALOEXCHANGE_FUSED_DATA_SETUP;

  switch ( vid ) {

    case Base_StdPar : {

      HALOEXCHANGE_FUSED_MANUAL_FUSER_SETUP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type pack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type  len  = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            pack_ptr_holders[pack_index] = ptr_holder{buffer, list, var};
            pack_lens[pack_index]        = len;
            pack_index += 1;
            buffer += len;
          }
        }

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), pack_index,
                         [=](Index_type j) {
          Real_ptr   buffer = pack_ptr_holders[j].buffer;
          Int_ptr    list   = pack_ptr_holders[j].list;
          Real_ptr   var    = pack_ptr_holders[j].var;
          Index_type len    = pack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            HALOEXCHANGE_FUSED_PACK_BODY;
          }
        });

        Index_type unpack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type  len  = unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            unpack_ptr_holders[unpack_index] = ptr_holder{buffer, list, var};
            unpack_lens[unpack_index]        = len;
            unpack_index += 1;
            buffer += len;
          }
        }

        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), unpack_index,
                         [=](Index_type j) {
          Real_ptr   buffer = unpack_ptr_holders[j].buffer;
          Int_ptr    list   = unpack_ptr_holders[j].list;
          Real_ptr   var    = unpack_ptr_holders[j].var;
          Index_type len    = unpack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            HALOEXCHANGE_FUSED_UNPACK_BODY;
          }
        });

      }
      stopTimer();

      HALOEXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN;

      break;
    }

    case Lambda_StdPar : {

      HALOEXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_SETUP;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Index_type pack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type  len  = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            new(&pack_lambdas[pack_index]) pack_lambda_type(make_pack_lambda(buffer, list, var));
            pack_lens[pack_index] = len;
            pack_index += 1;
            buffer += len;
          }
        }
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), pack_index,
                         [=](Index_type j) {
          auto       pack_lambda = pack_lambdas[j];
          Index_type len         = pack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            pack_lambda(i);
          }
        });

        Index_type unpack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type  len  = unpack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            new(&unpack_lambdas[unpack_index]) unpack_lambda_type(make_unpack_lambda(buffer, list, var));
            unpack_lens[unpack_index] = len;
            unpack_index += 1;
            buffer += len;
          }
        }
        std::for_each_n( std::execution::par_unseq,
                         counting_iterator<Index_type>(0), unpack_index,
                         [=](Index_type j) {
          auto       unpack_lambda = unpack_lambdas[j];
          Index_type len           = unpack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            unpack_lambda(i);
          }
        });

      }
      stopTimer();

      HALOEXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN;

      break;
    }

    default : {
      getCout() << "\n HALOEXCHANGE_FUSED : Unknown variant id = " << vid << std::endl;
    }

  }
#endif
}

} // end namespace apps
} // end namespace rajaperf
