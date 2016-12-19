/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing executor class that runs suite.
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


#ifndef RAJAPerfExecutor_HXX

#include "common/RunParams.hxx"

namespace rajaperf {

class Executor
{
public:
  Executor(RunParams& params);

  ~Executor();

  void runSuite();

private:
  Executor() = delete;

};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
