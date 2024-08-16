//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Type and value definitions to support Windows builds.
/// Windows has different syntax and does not support all POSIX modes
/// that Linux systems do.
///

#ifndef RAJAPerf_WindowsTypes_HPP
#define RAJAPerf_WindowsTypes_HPP

#include<sys/types.h>

#define _CRT_INTERNAL_NONSTDC_NAMES 1
#include<sys/stat.h>
#if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
  #define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

#include<filesystem>
#include<direct.h>
#include<io.h>

using mode_t = int;

/// Note: For the POSIX modes that do not have a Windows equivalent, the modes
///       defined here use the POSIX values left shifted 16 bits.

static constexpr mode_t S_IRUSR      = mode_t(_S_IREAD);   ///< read by user
static constexpr mode_t S_IWUSR      = mode_t(_S_IWRITE);  ///< write by user
static constexpr mode_t S_IXUSR      = 0x00400000;         ///< does nothing

static constexpr mode_t MS_MODE_MASK = 0x0000ffff;         ///< low word

#endif  // closing endif for header file include guard
