//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJAPerfSuite.hpp"
#include "OutputUtils.hpp"

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
#include <mpi.h>
#endif

#include<cstdlib>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>

#include<sys/types.h>
#include<sys/stat.h>


namespace rajaperf
{

/*
 * Recursively create directories for given path.
 */
std::string recursiveMkdir(const std::string& in_path)
{
  std::string path = in_path;

  // remove leading "." or "./"
  if ( !path.empty() ) {
    if ( path.at(0) == '.' ) {
      if ( path.length() > 2 && path.at(1) == '/' ) {
        path = in_path.substr(2, in_path.length()-2);
      } else {
        path = std::string();
      }
    }
  }

  if ( path.empty() ) return std::string();

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Processes wait for rank 0 to make the directories before proceeding
  if (rank != 0) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

// ----------------------------------------
  std::string outpath = path;

  mode_t mode = (S_IRUSR | S_IWUSR | S_IXUSR);
  const char separator = '/';

  int length = static_cast<int>(path.length());
  char* path_buf = new char[length + 1];
  sprintf(path_buf, "%s", path.c_str());
  struct stat status;
  int pos = length - 1;

  /* find part of path that has not yet been created */
  while ((stat(path_buf, &status) != 0) && (pos >= 0)) {

    /* slide backwards in string until next slash found */
    bool slash_found = false;
    while ((!slash_found) && (pos >= 0)) {
      if (path_buf[pos] == separator) {
        slash_found = true;
        if (pos >= 0) path_buf[pos] = '\0';
      } else pos--;
    }
  }

  /*
   * if there is a part of the path that already exists make sure
   * it is really a directory
   */
  if (pos >= 0) {
    if (!S_ISDIR(status.st_mode)) {
      getCout() << "Cannot create directories in path = " << path
                << "\n    because some intermediate item in path exists and"
                << "is NOT a directory" << std::endl;
       outpath = std::string();
    }
  }

  /*
   * make all directories that do not already exist
   *
   * if (pos < 0), then there is no part of the path that
   * already exists.  Need to make the first part of the
   * path before sliding along path_buf.
   */
  if ( !outpath.empty() && pos < 0) {
    if (mkdir(path_buf, mode) != 0) {
      getCout() << "   Cannot create directory  = "
                << path_buf << std::endl;
      outpath = std::string();
    }
    pos = 0;
  }

  if ( !outpath.empty() ) {

    /* make remaining directories */
    do {

      /* slide forward in string until next '\0' found */
         bool null_found = false;
      while ((!null_found) && (pos < length)) {
        if (path_buf[pos] == '\0') {
          null_found = true;
          path_buf[pos] = separator;
        }
        pos++;
      }

      /* make directory if not at end of path */
      if (pos < length) {
        if (mkdir(path_buf, mode) != 0) {
          getCout() << "   Cannot create directory  = "
                    << path_buf << std::endl;
          outpath = std::string();
        }
      }
    } while (pos < length && !outpath.empty());

  }

  delete[] path_buf;

#if defined(RAJA_PERFSUITE_ENABLE_MPI)
  // Rank 0 lets the other processes know it made the directories
  if (rank == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  return outpath;
}

}  // closing brace for rajaperf namespace
