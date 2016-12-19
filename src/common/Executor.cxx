/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for executor class that runs suite.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hxx"

#include "common/RAJAPerfSuite.hxx"

namespace rajaperf {

Executor::Executor(RunParams& params)
{
  for (int ikern = 0; ikern < NUM_KERNELS; ++ikern) {
     
  }
}

Executor::~Executor()
{
}

void Executor::runSuite()
{
}

}  // closing brace for rajaperf namespace
