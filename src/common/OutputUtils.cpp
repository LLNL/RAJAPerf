//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
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

#if defined(_WIN32) && 0
#include<io.h>

typedef int mode_t;

/// @Note If STRICT_UGO_PERMISSIONS is not defined, then setting Read for any
///       of User, Group, or Other will set Read for User and setting Write
///       will set Write for User.  Otherwise, Read and Write for Group and
///       Other are ignored.
///
/// @Note For the POSIX modes that do not have a Windows equivalent, the modes
///       defined here use the POSIX values left shifted 16 bits.

static const mode_t S_ISUID      = 0x08000000;           ///< does nothing
static const mode_t S_ISGID      = 0x04000000;           ///< does nothing
static const mode_t S_ISVTX      = 0x02000000;           ///< does nothing
static const mode_t S_IRUSR      = mode_t(_S_IREAD);     ///< read by user
static const mode_t S_IWUSR      = mode_t(_S_IWRITE);    ///< write by user
static const mode_t S_IXUSR      = 0x00400000;           ///< does nothing
#   ifndef STRICT_UGO_PERMISSIONS
static const mode_t S_IRGRP      = mode_t(_S_IREAD);     ///< read by *USER*
static const mode_t S_IWGRP      = mode_t(_S_IWRITE);    ///< write by *USER*
static const mode_t S_IXGRP      = 0x00080000;           ///< does nothing
static const mode_t S_IROTH      = mode_t(_S_IREAD);     ///< read by *USER*
static const mode_t S_IWOTH      = mode_t(_S_IWRITE);    ///< write by *USER*
static const mode_t S_IXOTH      = 0x00010000;           ///< does nothing
#   else
static const mode_t S_IRGRP      = 0x00200000;           ///< does nothing
static const mode_t S_IWGRP      = 0x00100000;           ///< does nothing
static const mode_t S_IXGRP      = 0x00080000;           ///< does nothing
static const mode_t S_IROTH      = 0x00040000;           ///< does nothing
static const mode_t S_IWOTH      = 0x00020000;           ///< does nothing
static const mode_t S_IXOTH      = 0x00010000;           ///< does nothing
#   endif

static const mode_t MS_MODE_MASK = 0x0000ffff;           ///< low word

#endif

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
#if defined(_WIN32) && 0
    if (_mkdir(path_buf, mode & MS_MODE_MASK) != 0) {
#else
    if (mkdir(path_buf, mode) != 0) {
#endif
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
#if defined(_WIN32) && 0
        if (_mkdir(path_buf, mode & MS_MODE_MASK) != 0) {
#else
        if (mkdir(path_buf, mode) != 0) {
#endif
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
