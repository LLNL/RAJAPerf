//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_RunParams_HPP
#define RAJAPerf_RunParams_HPP

#include <string>
#include <set>
#include <vector>
#include <array>
#include <iosfwd>

#include "RAJAPerfSuite.hpp"
#include "RPTypes.hpp"

namespace rajaperf
{

class InputParams;

/*!
 *******************************************************************************
 *
 * \brief Simple class to hold kernel execution parameters.
 *
 *******************************************************************************
 */
class RunParams {

public:
  explicit RunParams(InputParams const& input_params);
  ~RunParams();

  RunParams() = delete;
  RunParams(RunParams const&) = delete;
  RunParams& operator=(RunParams const&) = delete;
  RunParams(RunParams &&) = delete;
  RunParams& operator=(RunParams &&) = delete;

//@{
//! @name Getters/setters for processing input and run parameters

  int getReps() const { return checkrun_reps; }
  double getRepFactor() const { return rep_fact; }

  double getSize() const { return size; }
  double getSizeFactor() const { return size_factor; }
  Size_type getDataAlignment() const { return data_alignment; }

  int getGPUStream() const { return gpu_stream; }
  size_t numValidGPUBlockSize() const { return gpu_block_sizes.size(); }
  bool validGPUBlockSize(size_t block_size) const
  {
    for (size_t valid_block_size : gpu_block_sizes) {
      if (valid_block_size == block_size) {
        return true;
      }
    }
    return false;
  }
  size_t numValidAtomicReplication() const { return atomic_replications.size(); }
  bool validAtomicReplication(size_t atomic_replication) const
  {
    for (size_t valid_atomic_replication : atomic_replications) {
      if (valid_atomic_replication == atomic_replication) {
        return true;
      }
    }
    return false;
  }
  size_t numValidItemsPerThread() const { return items_per_threads.size(); }
  bool validItemsPerThread(size_t items_per_thread) const
  {
    for (size_t valid_items_per_thread : items_per_threads) {
      if (valid_items_per_thread == items_per_thread) {
        return true;
      }
    }
    return false;
  }

  int getMPISize() const { return mpi_size; }
  int getMPIRank() const { return mpi_rank; }
  bool validMPI3DDivision() const { return (mpi_3d_division[0]*mpi_3d_division[1]*mpi_3d_division[2] == mpi_size); }
  std::array<int, 3> const& getMPI3DDivision() const { return mpi_3d_division; }

  DataSpace getSeqDataSpace() const { return seqDataSpace; }
  DataSpace getOmpDataSpace() const { return ompDataSpace; }
  DataSpace getOmpTargetDataSpace() const { return ompTargetDataSpace; }
  DataSpace getCudaDataSpace() const { return cudaDataSpace; }
  DataSpace getHipDataSpace() const { return hipDataSpace; }
  DataSpace getKokkosDataSpace() const { return kokkosDataSpace; }

  DataSpace getSeqReductionDataSpace() const { return seqReductionDataSpace; }
  DataSpace getOmpReductionDataSpace() const { return ompReductionDataSpace; }
  DataSpace getOmpTargetReductionDataSpace() const { return ompTargetReductionDataSpace; }
  DataSpace getCudaReductionDataSpace() const { return cudaReductionDataSpace; }
  DataSpace getHipReductionDataSpace() const { return hipReductionDataSpace; }
  DataSpace getKokkosReductionDataSpace() const { return kokkosReductionDataSpace; }

  DataSpace getSeqMPIDataSpace() const { return seqMPIDataSpace; }
  DataSpace getOmpMPIDataSpace() const { return ompMPIDataSpace; }
  DataSpace getOmpTargetMPIDataSpace() const { return ompTargetMPIDataSpace; }
  DataSpace getCudaMPIDataSpace() const { return cudaMPIDataSpace; }
  DataSpace getHipMPIDataSpace() const { return hipMPIDataSpace; }
  DataSpace getKokkosMPIDataSpace() const { return kokkosMPIDataSpace; }

//@}

  /*!
   * \brief Print all run params data to given output stream.
   */
  void print(std::ostream& str) const;


private:
  int checkrun_reps;     /*!< Num reps each kernel is run in check run */
  double rep_fact;       /*!< pct of default kernel reps to run */

  double size;           /*!< kernel size to run (input option) */
  double size_factor;    /*!< default kernel size multipier (input option) */
  Size_type data_alignment;

  int gpu_stream; /*!< 0 -> use stream 0; anything else -> use raja default stream */
  std::vector<size_t> gpu_block_sizes; /*!< Block sizes for gpu tunings to run (input option) */
  std::vector<size_t> atomic_replications; /*!< Atomic replications for gpu tunings to run (input option) */
  std::vector<size_t> items_per_threads; /*!< Items per thread for gpu tunings to run (input option) */

  int mpi_size;           /*!< Number of MPI ranks */
  int mpi_rank;           /*!< Rank of this MPI process */
  std::array<int, 3> mpi_3d_division; /*!< Number of MPI ranks in each dimension of a 3D grid */

  DataSpace seqDataSpace;
  DataSpace ompDataSpace;
  DataSpace ompTargetDataSpace;
  DataSpace cudaDataSpace;
  DataSpace hipDataSpace;
  DataSpace kokkosDataSpace;

  DataSpace seqReductionDataSpace;
  DataSpace ompReductionDataSpace;
  DataSpace ompTargetReductionDataSpace;
  DataSpace cudaReductionDataSpace;
  DataSpace hipReductionDataSpace;
  DataSpace kokkosReductionDataSpace;

  DataSpace seqMPIDataSpace;
  DataSpace ompMPIDataSpace;
  DataSpace ompTargetMPIDataSpace;
  DataSpace cudaMPIDataSpace;
  DataSpace hipMPIDataSpace;
  DataSpace kokkosMPIDataSpace;
};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
