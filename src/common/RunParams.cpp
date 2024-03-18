//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RunParams.hpp"

#include "InputParams.hpp"

#include <cstdlib>
#include <cstdio>
#include <iostream>

namespace rajaperf
{

/*
 *******************************************************************************
 *
 * Ctor for RunParams class copies arguments from input_params.
 *
 *******************************************************************************
 */
RunParams::RunParams(InputParams const& input_params)
 : checkrun_reps(input_params.getReps()),
   rep_fact(input_params.getRepFactor()),

   size(input_params.getSize()),
   size_factor(input_params.getSizeFactor()),
   data_alignment(input_params.getDataAlignment()),

   gpu_stream(input_params.getGPUStream()),
   gpu_block_sizes(input_params.getGPUBlockSizes()),
   atomic_replications(input_params.getAtomicReplications()),
   items_per_threads(input_params.getItemsPerThreads()),

   mpi_size(input_params.getMPISize()),
   mpi_rank(input_params.getMPIRank()),
   mpi_3d_division(input_params.getMPI3DDivision()),

   seqDataSpace(input_params.getSeqDataSpace()),
   ompDataSpace(input_params.getOmpDataSpace()),
   ompTargetDataSpace(input_params.getOmpTargetDataSpace()),
   cudaDataSpace(input_params.getCudaDataSpace()),
   hipDataSpace(input_params.getHipDataSpace()),
   kokkosDataSpace(input_params.getKokkosDataSpace()),

   seqReductionDataSpace(input_params.getSeqReductionDataSpace()),
   ompReductionDataSpace(input_params.getOmpReductionDataSpace()),
   ompTargetReductionDataSpace(input_params.getOmpTargetReductionDataSpace()),
   cudaReductionDataSpace(input_params.getCudaReductionDataSpace()),
   hipReductionDataSpace(input_params.getHipReductionDataSpace()),
   kokkosReductionDataSpace(input_params.getKokkosReductionDataSpace()),

   seqMPIDataSpace(input_params.getSeqMPIDataSpace()),
   ompMPIDataSpace(input_params.getOmpMPIDataSpace()),
   ompTargetMPIDataSpace(input_params.getOmpTargetMPIDataSpace()),
   cudaMPIDataSpace(input_params.getCudaMPIDataSpace()),
   hipMPIDataSpace(input_params.getHipMPIDataSpace()),
   kokkosMPIDataSpace(input_params.getKokkosMPIDataSpace())
{
}


/*
 *******************************************************************************
 *
 * Dtor for RunParams class.
 *
 *******************************************************************************
 */
RunParams::~RunParams()
{
}


/*
 *******************************************************************************
 *
 * Print all kernel params data to given output stream.
 *
 *******************************************************************************
 */
void RunParams::print(std::ostream& str) const
{
  str << "\n checkrun_reps = " << checkrun_reps;
  str << "\n rep_fact = " << rep_fact;

  str << "\n size = " << size;
  str << "\n size_factor = " << size_factor;
  str << "\n data_alignment = " << data_alignment;

  str << "\n gpu stream = " << ((gpu_stream == 0) ? "0" : "RAJA default");
  str << "\n gpu_block_sizes = ";
  for (size_t j = 0; j < gpu_block_sizes.size(); ++j) {
    str << "\n\t" << gpu_block_sizes[j];
  }
  str << "\n atomic_replications = ";
  for (size_t j = 0; j < atomic_replications.size(); ++j) {
    str << "\n\t" << atomic_replications[j];
  }
  str << "\n items_per_threads = ";
  for (size_t j = 0; j < items_per_threads.size(); ++j) {
    str << "\n\t" << items_per_threads[j];
  }

  str << "\n mpi_size = " << mpi_size;
  str << "\n mpi_3d_division = ";
  for (size_t j = 0; j < 3; ++j) {
    str << "\n\t" << mpi_3d_division[j];
  }

  str << "\n seq data space = " << getDataSpaceName(seqDataSpace);
  str << "\n omp data space = " << getDataSpaceName(ompDataSpace);
  str << "\n omp target data space = " << getDataSpaceName(ompTargetDataSpace);
  str << "\n cuda data space = " << getDataSpaceName(cudaDataSpace);
  str << "\n hip data space = " << getDataSpaceName(hipDataSpace);
  str << "\n kokkos data space = " << getDataSpaceName(kokkosDataSpace);

  str << "\n seq reduction data space = " << getDataSpaceName(seqReductionDataSpace);
  str << "\n omp reduction data space = " << getDataSpaceName(ompReductionDataSpace);
  str << "\n omp target reduction data space = " << getDataSpaceName(ompTargetReductionDataSpace);
  str << "\n cuda reduction data space = " << getDataSpaceName(cudaReductionDataSpace);
  str << "\n hip reduction data space = " << getDataSpaceName(hipReductionDataSpace);
  str << "\n kokkos reduction data space = " << getDataSpaceName(kokkosReductionDataSpace);

  str << "\n seq MPI data space = " << getDataSpaceName(seqMPIDataSpace);
  str << "\n omp MPI data space = " << getDataSpaceName(ompMPIDataSpace);
  str << "\n omp target MPI data space = " << getDataSpaceName(ompTargetMPIDataSpace);
  str << "\n cuda MPI data space = " << getDataSpaceName(cudaMPIDataSpace);
  str << "\n hip MPI data space = " << getDataSpaceName(hipMPIDataSpace);
  str << "\n kokkos MPI data space = " << getDataSpaceName(kokkosMPIDataSpace);

  str << std::endl;
  str.flush();
}

}  // closing brace for rajaperf namespace
