//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HALO_EXCHANGE_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace comm
{


void HALO_EXCHANGE_FUSED::runOpenMPVariantDirect(VariantID vid)
{

  const Index_type run_reps = getRunReps();

  HALO_EXCHANGE_FUSED_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      HALO_EXCHANGE_FUSED_MANUAL_FUSER_SETUP;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        Index_type pack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            pack_ptr_holders[pack_index] = ptr_holder{buffer, list, var};
            pack_lens[pack_index] = len;
            pack_index += 1;
            buffer += len;
          }
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < pack_index; j++) {
          #pragma omp task firstprivate(j)
          {
            Real_ptr   buffer = pack_ptr_holders[j].buffer;
            Int_ptr    list   = pack_ptr_holders[j].list;
            Real_ptr   var    = pack_ptr_holders[j].var;
            Index_type len    = pack_lens[j];
            for (Index_type i = 0; i < len; i++) {
              HALO_PACK_BODY;
            }
          }
        }
#else
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_pack");
        #pragma omp parallel for
        for (Index_type j = 0; j < pack_index; j++) {
          Real_ptr   buffer = pack_ptr_holders[j].buffer;
          Int_ptr    list   = pack_ptr_holders[j].list;
          Real_ptr   var    = pack_ptr_holders[j].var;
          Index_type len    = pack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            HALO_PACK_BODY;
          }
        }
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_pack");
#endif
        if (separate_buffers) {
          for (Index_type l = 0; l < num_neighbors; ++l) {
            Index_type len = pack_index_list_lengths[l];
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

        Index_type unpack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            unpack_ptr_holders[unpack_index] = ptr_holder{buffer, list, var};
            unpack_lens[unpack_index] = len;
            unpack_index += 1;
            buffer += len;
          }
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < unpack_index; j++) {
          #pragma omp task firstprivate(j)
          {
            Real_ptr   buffer = unpack_ptr_holders[j].buffer;
            Int_ptr    list   = unpack_ptr_holders[j].list;
            Real_ptr   var    = unpack_ptr_holders[j].var;
            Index_type len    = unpack_lens[j];
            for (Index_type i = 0; i < len; i++) {
              HALO_UNPACK_BODY;
            }
          }
        }
#else
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_unpack");
        #pragma omp parallel for
        for (Index_type j = 0; j < unpack_index; j++) {
          Real_ptr   buffer = unpack_ptr_holders[j].buffer;
          Int_ptr    list   = unpack_ptr_holders[j].list;
          Real_ptr   var    = unpack_ptr_holders[j].var;
          Index_type len    = unpack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            HALO_UNPACK_BODY;
          }
        }
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_unpack");
#endif

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      HALO_EXCHANGE_FUSED_MANUAL_FUSER_TEARDOWN;

      break;
    }

    case Lambda_OpenMP : {

      HALO_EXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_SETUP;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        Index_type pack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            new(&pack_lambdas[pack_index]) pack_lambda_type(make_pack_lambda(buffer, list, var));
            pack_lens[pack_index] = len;
            pack_index += 1;
            buffer += len;
          }
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < pack_index; j++) {
          #pragma omp task firstprivate(j)
          {
            auto       pack_lambda = pack_lambdas[j];
            Index_type len         = pack_lens[j];
            for (Index_type i = 0; i < len; i++) {
              pack_lambda(i);
            }
          }
        }
#else
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_pack");
        #pragma omp parallel for
        for (Index_type j = 0; j < pack_index; j++) {
          auto       pack_lambda = pack_lambdas[j];
          Index_type len         = pack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            pack_lambda(i);
          }
        }
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_pack");
#endif
        if (separate_buffers) {
          for (Index_type l = 0; l < num_neighbors; ++l) {
            Index_type len = pack_index_list_lengths[l];
            memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

        Index_type unpack_index = 0;

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            new(&unpack_lambdas[unpack_index]) unpack_lambda_type(make_unpack_lambda(buffer, list, var));
            unpack_lens[unpack_index] = len;
            unpack_index += 1;
            buffer += len;
          }
        }

#if defined(RAJA_ENABLE_OMP_TASK_INTERNAL)
        #pragma omp parallel
        #pragma omp single nowait
        for (Index_type j = 0; j < unpack_index; j++) {
          #pragma omp task firstprivate(j)
          {
            auto       unpack_lambda = unpack_lambdas[j];
            Index_type len           = unpack_lens[j];
            for (Index_type i = 0; i < len; i++) {
              unpack_lambda(i);
            }
          }
        }
#else
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_unpack");
        #pragma omp parallel for
        for (Index_type j = 0; j < unpack_index; j++) {
          auto       unpack_lambda = unpack_lambdas[j];
          Index_type len           = unpack_lens[j];
          for (Index_type i = 0; i < len; i++) {
            unpack_lambda(i);
          }
        }
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_unpack");
#endif

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      HALO_EXCHANGE_FUSED_MANUAL_LAMBDA_FUSER_TEARDOWN;

      break;
    }

    default : {
      getCout() << "\n HALO_EXCHANGE_FUSED : Unknown variant id = " << vid << std::endl;
    }

  }

}

template < typename dispatch_helper >
void HALO_EXCHANGE_FUSED::runOpenMPVariantWorkGroup(VariantID vid)
{

  const Index_type run_reps = getRunReps();

  HALO_EXCHANGE_FUSED_DATA_SETUP;

  switch ( vid ) {

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      using AllocatorHolder = RAJAPoolAllocatorHolder<
        RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator>>;
      using Allocator = AllocatorHolder::Allocator<char>;

      AllocatorHolder allocatorHolder;

      using range_segment = RAJA::TypedRangeSegment<Index_type>;

      using dispatch_policy = typename dispatch_helper::template dispatch_policy<
                                camp::list<range_segment, Packer>,
                                camp::list<range_segment, UnPacker>>;

      using workgroup_policy = RAJA::WorkGroupPolicy <
                                   RAJA::omp_work,
                                   RAJA::ordered,
                                   RAJA::constant_stride_array_of_objects,
                                   dispatch_policy >;

      using workpool = RAJA::WorkPool< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

      using workgroup = RAJA::WorkGroup< workgroup_policy,
                                         Index_type,
                                         RAJA::xargs<>,
                                         Allocator >;

      using worksite = RAJA::WorkSite< workgroup_policy,
                                       Index_type,
                                       RAJA::xargs<>,
                                       Allocator >;

      workpool pool_pack  (allocatorHolder.template getAllocator<char>());
      workpool pool_unpack(allocatorHolder.template getAllocator<char>());
      pool_pack.reserve(num_neighbors * num_vars, 1024ull*1024ull);
      pool_unpack.reserve(num_neighbors * num_vars, 1024ull*1024ull);

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = unpack_index_list_lengths[l];
          MPI_Irecv(recv_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], recv_tags[l], MPI_COMM_WORLD, &unpack_mpi_requests[l]);
        }

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = pack_buffers[l];
          Int_ptr list = pack_index_lists[l];
          Index_type len = pack_index_list_lengths[l];
          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            pool_pack.enqueue(range_segment(0, len), Packer{buffer, var, list});
            buffer += len;
          }
        }
        workgroup group_pack = pool_pack.instantiate();
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_pack");
        worksite site_pack = group_pack.run(res);
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_pack");
        if (separate_buffers) {
          for (Index_type l = 0; l < num_neighbors; ++l) {
            Index_type len = pack_index_list_lengths[l];
            res.memcpy(send_buffers[l], pack_buffers[l], len*num_vars*sizeof(Real_type));
          }
        }
        for (Index_type l = 0; l < num_neighbors; ++l) {
          Index_type len = pack_index_list_lengths[l];
          MPI_Isend(send_buffers[l], len*num_vars, Real_MPI_type,
              mpi_ranks[l], send_tags[l], MPI_COMM_WORLD, &pack_mpi_requests[l]);
        }

        MPI_Waitall(num_neighbors, unpack_mpi_requests.data(), MPI_STATUSES_IGNORE);

        for (Index_type l = 0; l < num_neighbors; ++l) {
          Real_ptr buffer = unpack_buffers[l];
          Int_ptr list = unpack_index_lists[l];
          Index_type len = unpack_index_list_lengths[l];
          if (separate_buffers) {
            res.memcpy(unpack_buffers[l], recv_buffers[l], len*num_vars*sizeof(Real_type));
          }

          for (Index_type v = 0; v < num_vars; ++v) {
            Real_ptr var = vars[v];
            pool_unpack.enqueue(range_segment(0, len), UnPacker{buffer, var, list});
            buffer += len;
          }
        }
        workgroup group_unpack = pool_unpack.instantiate();
        RP_CALI_SUBKERNEL_BEGIN("HALO_EXCHANGE_FUSED_unpack");
        worksite site_unpack = group_unpack.run(res);
        RP_CALI_SUBKERNEL_END("HALO_EXCHANGE_FUSED_unpack");

        MPI_Waitall(num_neighbors, pack_mpi_requests.data(), MPI_STATUSES_IGNORE);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n HALO_EXCHANGE_FUSED : Unknown variant id = " << vid << std::endl;
    }

  }

}


void HALO_EXCHANGE_FUSED::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    if (vid == Base_OpenMP || vid == Lambda_OpenMP) {

      addVariantTuning<&HALO_EXCHANGE_FUSED::runOpenMPVariantDirect>(
          vid, "direct");

    }

    if (vid == RAJA_OpenMP) {

      seq_for(workgroup_dispatch_helpers{}, [&](auto dispatch_helper) {

        addVariantTuning<&HALO_EXCHANGE_FUSED::runOpenMPVariantWorkGroup<
                             decltype(dispatch_helper)>>(
            vid, decltype(dispatch_helper)::get_name());

      });

    }

  }

}

} // end namespace comm
} // end namespace rajaperf

#endif

#endif
