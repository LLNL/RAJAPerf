//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>


namespace RAJA
{
namespace statement
{

/*!
 * A RAJA::kernel statement that implements a tiling (or blocking) loop.
 *
 */
template <camp::idx_t ArgumentId,
          typename TilePolicy,
          typename ExecPolicy,
          typename... EnclosedStmts>
struct TileExp : public internal::Statement<ExecPolicy, EnclosedStmts...> {
  using tile_policy_t = TilePolicy;
  using exec_policy_t = ExecPolicy;
};

/*!
 * A RAJA::kernel statement that implements a tiled loop.
 *
 */
template <camp::idx_t ArgumentId,
          typename TilePolicy,
          typename TileExecPolicy,
          typename ForExecPolicy,
          typename... EnclosedStmts>
struct TiledFor : public internal::Statement<TileExecPolicy, EnclosedStmts...> {
  using tile_policy_t = TilePolicy;
  using exec_policy_t = TileExecPolicy;
};

}  // end namespace statement

namespace internal
{
/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileExp
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileExp<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    hip_block_xyz_direct<BlockDim>,
                    EnclosedStmts...>,
                    Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this TileExp statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    diff_t len = segment.end() - segment.begin();
    diff_t i = get_hip_dim<BlockDim>(dim3(blockIdx.x,blockIdx.y,blockIdx.z)) * chunk_size;

    // Keep copy of original segment, so we can restore it
    segment_t orig_segment = segment;

    // Assign our new tiled segment
    segment = orig_segment.slice(i, chunk_size);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && (i<len));

    // Set range back to original values
    segment = orig_segment;
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // Compute how many blocks
    diff_t len = segment_length<ArgumentId>(data);
    diff_t num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len) {
      num_blocks++;
    }

    LaunchDims dims;
    set_hip_dim<BlockDim>(dims.blocks, num_blocks);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<BlockDim>(dims.min_blocks, num_blocks);


    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};


/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TiledFor
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int BlockDim,
          int ThreadDim,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::TiledFor<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    hip_block_xyz_direct<BlockDim>,
                    hip_thread_xyz_direct<ThreadDim>,
                    EnclosedStmts...>,
                    Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i = get_hip_dim<BlockDim>(dim3(blockIdx.x,blockIdx.y,blockIdx.z)) * chunk_size +
               get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute how many blocks
    diff_t len = segment_length<ArgumentId>(data);
    diff_t num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len) {
      num_blocks++;
    }

    LaunchDims dims;
    set_hip_dim<BlockDim>(dims.blocks, num_blocks);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<BlockDim>(dims.min_blocks, num_blocks);

    // add a max_blocks to ensure correctness?


    set_hip_dim<ThreadDim>(dims.threads, chunk_size);

    // since we are direct-mapping, we REQUIRE chunk_size
    set_hip_dim<ThreadDim>(dims.min_threads, chunk_size);


    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

}
}


namespace rajaperf
{
namespace basic
{

  //
  // Define thread block shape for Hip execution
  //
#define i_block_sz (32)
#define j_block_sz (block_size / i_block_sz)
#define k_block_sz (1)

#define NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  i_block_sz, j_block_sz, k_block_sz

#define NESTED_INIT_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP); \
  static_assert(i_block_sz*j_block_sz*k_block_sz == block_size, "Invalid block_size");

#define NESTED_INIT_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, i_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nk, k_block_sz)));


#define NESTED_INIT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(array, m_array, m_array_length);

#define NESTED_INIT_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_array, array, m_array_length); \
  deallocHipDeviceData(array);

template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
  __launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    NESTED_INIT_BODY;
  }
}

template< size_t i_block_size, size_t j_block_size, size_t k_block_size, typename Lambda >
__launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_lam(Index_type ni, Index_type nj, Index_type nk,
                                Lambda body)
{
  Index_type i = blockIdx.x * i_block_size + threadIdx.x;
  Index_type j = blockIdx.y * j_block_size + threadIdx.y;
  Index_type k = blockIdx.z;

  if ( i < ni && j < nj && k < nk ) {
    body(i, j, k);
  }
}

// implementation essentially an lining of this RAJA::kernel policy
// using EXEC_POL =
//   RAJA::KernelPolicy<
//    RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
//       RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
//                                RAJA::hip_block_y_direct,
//         RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
//                                  RAJA::hip_block_x_direct,
//           RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
//             RAJA::statement::For<1, RAJA::hip_thread_y_direct,   // j
//               RAJA::statement::For<0, RAJA::hip_thread_x_direct, // i
//                 RAJA::statement::Lambda<0>
//               >
//             >
//           >
//         >
//       >
//     >
//   >;
template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
  __launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_exp0(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  bool thread_active0 = true;
  Index_type i, j, k;

  // RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>, RAJA::hip_block_y_direct,
  j = blockIdx.y * static_cast<Index_type>(j_block_size);
  if (j < nj) {

    Index_type tile_nj = min(j+static_cast<Index_type>(j_block_size), nj);

    // RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>, RAJA::hip_block_x_direct,
    i = blockIdx.x * static_cast<Index_type>(i_block_size);
    if (i < ni) {

      Index_type tile_ni = min(i+static_cast<Index_type>(i_block_size), ni);

      // RAJA::statement::For<2, RAJA::hip_block_z_direct,
      {
        k = blockIdx.z;

        bool thread_active1 = thread_active0 && (k<nk);

        // RAJA::statement::For<1, RAJA::hip_thread_y_direct,
        {
          j = j + threadIdx.y;

          bool thread_active2 = thread_active1 && (j<tile_nj);

          // RAJA::statement::For<0, RAJA::hip_thread_x_direct,
          {
            i = i + threadIdx.x;

            bool thread_active3 = thread_active2 && (i<tile_ni);

            // RAJA::statement::Lambda<0>
            if ( thread_active3 ) {
              NESTED_INIT_BODY;
            }

          }

        }

      }

    }

  }

}


template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
  __launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_exp1(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  bool thread_active0 = true;
  Index_type i, j, k;

  // RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>, RAJA::hip_block_y_direct,
  j = blockIdx.y * static_cast<Index_type>(j_block_size);
  {

    Index_type tile_nj = min(j+static_cast<Index_type>(j_block_size), nj);

    // RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>, RAJA::hip_block_x_direct,
    i = blockIdx.x * static_cast<Index_type>(i_block_size);
    {

      Index_type tile_ni = min(i+static_cast<Index_type>(i_block_size), ni);

      // RAJA::statement::For<2, RAJA::hip_block_z_direct,
      {
        k = blockIdx.z;

        bool thread_active1 = thread_active0 && (k<nk);

        // RAJA::statement::For<1, RAJA::hip_thread_y_direct,
        {
          j = j + threadIdx.y;

          bool thread_active2 = thread_active1 && (j<tile_nj);

          // RAJA::statement::For<0, RAJA::hip_thread_x_direct,
          {
            i = i + threadIdx.x;

            bool thread_active3 = thread_active2 && (i<tile_ni);

            // RAJA::statement::Lambda<0>
            if ( thread_active3 ) {
              NESTED_INIT_BODY;
            }

          }

        }

      }

    }

  }

}


template< size_t i_block_size, size_t j_block_size, size_t k_block_size >
  __launch_bounds__(i_block_size*j_block_size*k_block_size)
__global__ void nested_init_exp2(Real_ptr array,
                            Index_type ni, Index_type nj, Index_type nk)
{
  bool thread_active0 = true;
  Index_type i, j, k;

  // RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>, RAJA::hip_block_y_direct,
  j = blockIdx.y * static_cast<Index_type>(j_block_size);
  {

    // RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>, RAJA::hip_block_x_direct,
    i = blockIdx.x * static_cast<Index_type>(i_block_size);
    {

      // RAJA::statement::For<2, RAJA::hip_block_z_direct,
      {
        k = blockIdx.z;

        bool thread_active1 = thread_active0 && (k<nk);

        // RAJA::statement::For<1, RAJA::hip_thread_y_direct,
        {
          j = j + threadIdx.y;

          bool thread_active2 = thread_active1 && (j<nj);

          // RAJA::statement::For<0, RAJA::hip_thread_x_direct,
          {
            i = i + threadIdx.x;

            bool thread_active3 = thread_active2 && (i<ni);

            // RAJA::statement::Lambda<0>
            if ( thread_active3 ) {
              NESTED_INIT_BODY;
            }

          }

        }

      }

    }

  }

}


template < size_t block_size >
void NESTED_INIT::runHipVariantExp(VariantID vid, size_t exp)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_HIP && (exp == 0) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;

      hipLaunchKernelGGL((nested_init_exp0<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         array, ni, nj, nk);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && (exp == 1) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;

      hipLaunchKernelGGL((nested_init_exp1<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         array, ni, nj, nk);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == Base_HIP && (exp == 2) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;

      hipLaunchKernelGGL((nested_init_exp2<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         array, ni, nj, nk);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP && (exp == 0) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    using EXEC_POL =
      RAJA::KernelPolicy<
       RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::TileExp<1, RAJA::tile_fixed<j_block_sz>,
                                   RAJA::hip_block_y_direct,
            RAJA::statement::TileExp<0, RAJA::tile_fixed<i_block_sz>,
                                     RAJA::hip_block_x_direct,
              RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
                RAJA::statement::For<1, RAJA::hip_thread_y_direct,   // j
                  RAJA::statement::For<0, RAJA::hip_thread_x_direct, // i
                    RAJA::statement::Lambda<0>
                  >
                >
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP && (exp == 1) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    using EXEC_POL =
      RAJA::KernelPolicy<
       RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::For<2, RAJA::hip_block_z_direct,              // k
            RAJA::statement::TiledFor<1, RAJA::tile_fixed<j_block_sz>,   // j
                RAJA::hip_block_y_direct, RAJA::hip_thread_y_direct,
              RAJA::statement::TiledFor<0, RAJA::tile_fixed<i_block_sz>, // i
                  RAJA::hip_block_x_direct, RAJA::hip_thread_x_direct,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP && (exp == 2) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    constexpr bool async = true;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<async, i_block_sz*j_block_sz*k_block_sz>>;

    using teams_x = RAJA::expt::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_y = RAJA::expt::LoopPolicy<RAJA::hip_block_y_direct>;
    using teams_z = RAJA::expt::LoopPolicy<RAJA::hip_block_z_direct>;

    using threads_x = RAJA::expt::LoopPolicy<RAJA::hip_thread_x_direct>;
    using threads_y = RAJA::expt::LoopPolicy<RAJA::hip_thread_y_direct>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type Bi = RAJA_DIVIDE_CEILING_INT(ni, i_block_sz);
      Index_type Bj = RAJA_DIVIDE_CEILING_INT(nj, j_block_sz);
      Index_type Bk = RAJA_DIVIDE_CEILING_INT(nk, k_block_sz);

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::Grid(RAJA::expt::Teams(Bi, Bj, Bk),
                         RAJA::expt::Threads(i_block_sz, j_block_sz, k_block_sz)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<teams_z>(ctx, RAJA::RangeSegment(0, Bk), [&](Index_type k) {
            RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Bj), [&](Index_type by) {
              RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Bi), [&](Index_type bx) {

                Index_type j_begin = by * j_block_sz;
                Index_type j_end = min(j_begin + j_block_sz, nj);
                Index_type i_begin = bx * i_block_sz;
                Index_type i_end = min(i_begin + i_block_sz, ni);
                RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(j_begin, j_end), [&](Index_type j) {
                  RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(i_begin, i_end), [&](Index_type i) {
                    NESTED_INIT_BODY;
                  });  // RAJA::expt::loop<threads_x>
                });  // RAJA::expt::loop<threads_y>

              });  // RAJA::expt::loop<teams_x>
            });  // RAJA::expt::loop<teams_y>
          });  // RAJA::expt::loop<teams_z>

      });  // RAJA::expt::launch

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP && (exp == 3) ) {

    NESTED_INIT_DATA_SETUP_HIP;

    constexpr bool async = true;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<async, i_block_sz*j_block_sz*k_block_sz>>;

    using teams_x = RAJA::expt::LoopPolicy<RAJA::hip_block_x_direct>;
    using teams_y = RAJA::expt::LoopPolicy<RAJA::hip_block_y_direct>;
    using teams_z = RAJA::expt::LoopPolicy<RAJA::hip_block_z_direct>;

    using threads_x = RAJA::expt::LoopPolicy<RAJA::hip_thread_x_direct>;
    using threads_y = RAJA::expt::LoopPolicy<RAJA::hip_thread_y_direct>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Index_type Bi = RAJA_DIVIDE_CEILING_INT(ni, i_block_sz);
      Index_type Bj = RAJA_DIVIDE_CEILING_INT(nj, j_block_sz);
      Index_type Bk = RAJA_DIVIDE_CEILING_INT(nk, k_block_sz);

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::Grid(RAJA::expt::Teams(Bi, Bj, Bk),
                         RAJA::expt::Threads(i_block_sz, j_block_sz, k_block_sz)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<teams_z>(ctx, RAJA::RangeSegment(0, Bk), [&](Index_type bz) {
            RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Bj), [&](Index_type by) {
              RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Bi), [&](Index_type bx) {

                RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, j_block_sz), [&](Index_type ty) {
                  RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, i_block_sz), [&](Index_type tx) {

                    Index_type i = bx * i_block_sz + tx;
                    Index_type j = by * j_block_sz + ty;
                    Index_type k = bz;

                    if ( i < ni && j < nj && k < nk ) {
                      NESTED_INIT_BODY;
                    }

                  });  // RAJA::expt::loop<threads_x>
                });  // RAJA::expt::loop<threads_y>

              });  // RAJA::expt::loop<teams_x>
            });  // RAJA::expt::loop<teams_y>
          });  // RAJA::expt::loop<teams_z>

      });  // RAJA::expt::launch

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}


template < size_t block_size >
void NESTED_INIT::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;

      hipLaunchKernelGGL((nested_init<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         array, ni, nj, nk);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      NESTED_INIT_THREADS_PER_BLOCK_HIP;
      NESTED_INIT_NBLOCKS_HIP;

      auto nested_init_lambda = [=] __device__ (Index_type i, Index_type j,
                                                Index_type k) {
        NESTED_INIT_BODY;
      };

      hipLaunchKernelGGL((nested_init_lam<NESTED_INIT_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(nested_init_lambda) >),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         ni, nj, nk, nested_init_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    NESTED_INIT_DATA_SETUP_HIP;

    using EXEC_POL =
      RAJA::KernelPolicy<
       RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
          RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>,
                                   RAJA::hip_block_y_direct,
            RAJA::statement::Tile<0, RAJA::tile_fixed<i_block_sz>,
                                     RAJA::hip_block_x_direct,
              RAJA::statement::For<2, RAJA::hip_block_z_direct,      // k
                RAJA::statement::For<1, RAJA::hip_thread_y_direct,   // j
                  RAJA::statement::For<0, RAJA::hip_thread_x_direct, // i
                    RAJA::statement::Lambda<0>
                  >
                >
              >
            >
          >
        >
      >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                               RAJA::RangeSegment(0, nj),
                                               RAJA::RangeSegment(0, nk)),
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
        NESTED_INIT_BODY;
      });

    }
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  NESTED_INIT : Unknown Hip variant id = " << vid << std::endl;
  }
}


void NESTED_INIT::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runHipVariantBlock<block_size>(vid);

      }

      t += 1;

    }

  });

  size_t num_exp = (vid == Base_HIP)   ? 3
                 : (vid == Lambda_HIP) ? 0
                 : (vid == RAJA_HIP)   ? 4
                 :                       0 ;
  for (size_t exp = 0; exp < num_exp; ++exp) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantExp<block_size>(vid, exp);

        }

        t += 1;

      }

    });

  }

}

void NESTED_INIT::setHipTuningDefinitions(VariantID vid)
{

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

    }

  });

  size_t num_exp = (vid == Base_HIP)   ? 3
                 : (vid == Lambda_HIP) ? 0
                 : (vid == RAJA_HIP)   ? 4
                 :                       0 ;
  for (size_t exp = 0; exp < num_exp; ++exp) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "exp"+std::to_string(exp)+"_"+std::to_string(block_size));

      }

    });

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
