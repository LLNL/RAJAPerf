//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_AppsData_HPP
#define RAJAPerf_AppsData_HPP

#include <ostream>

#include "common/RPTypes.hpp"

namespace rajaperf
{
namespace apps
{

//
// Some macros used in kernels to mimic real app code style.
//
#define NDPTRSET(jp, kp,v,v0,v1,v2,v3,v4,v5,v6,v7)  \
   v0 = v ;   \
   v1 = v0 + 1 ;  \
   v2 = v0 + jp ; \
   v3 = v1 + jp ; \
   v4 = v0 + kp ; \
   v5 = v1 + kp ; \
   v6 = v2 + kp ; \
   v7 = v3 + kp ;

#define NDSET2D(jp,v,v1,v2,v3,v4)  \
   v4 = v ;   \
   v1 = v4 + 1 ;  \
   v2 = v1 + jp ;  \
   v3 = v4 + jp ;

#define zabs2(z)    ( real(z)*real(z)+imag(z)*imag(z) )


//
// Domain structure to mimic structured mesh loops code style.
//
class ADomain
{
public:

   ADomain() = delete;

   ADomain( Index_type real_nodes_per_dim, Index_type ndims )
      : ndims(ndims), NPNL(2), NPNR(1)
   {
      int NPZL = NPNL - 1;
      int NPZR = NPNR+1 - 1;

      if ( ndims >= 1 ) {
         imin = NPNL;
         imax = NPNL + real_nodes_per_dim-1;
         nnalls = (imax+1 - imin + NPNL + NPNR);
         n_real_zones = (imax - imin);
         n_real_nodes = (imax+1 - imin);
      } else {
         imin = 0;
         imax = 0;
         nnalls = 0;
      }

      if ( ndims >= 2 ) {
         jmin = NPNL;
         jmax = NPNL + real_nodes_per_dim-1;
         jp = nnalls;
         nnalls *= (jmax+1 - jmin + NPNL + NPNR);
         n_real_zones *= (jmax - jmin);
         n_real_nodes *= (jmax+1 - jmin);
      } else {
         jmin = 0;
         jmax = 0;
         jp = 0;
      }

      if ( ndims >= 3 ) {
         kmin = NPNL;
         kmax = NPNL + real_nodes_per_dim-1;
         kp = nnalls;
         nnalls *= (kmax+1 - kmin + NPNL + NPNR);
         n_real_zones *= (kmax - kmin);
         n_real_nodes *= (kmax+1 - kmin);
      } else {
         kmin = 0;
         kmax = 0;
         kp = 0;
      }

      frn = kmin*kp + jmin*jp + imin;
      lrn = kmax*kp + jmax*jp + imax;
      fpn = (kmin - NPNL)*kp + (jmin - NPNL)*jp + (imin - NPNL);
      lpn = (kmax + NPNR)*kp + (jmax + NPNR)*jp + (imax + NPNR);

      fpz = (kmin - NPZL)*kp + (jmin - NPZL)*jp + (imin - NPZL);
      lpz = (kmax-1 + NPZR)*kp + (jmax-1 + NPZR)*jp + (imax-1 + NPZR);
   }

   ~ADomain()
   {
   }

   Index_type ndims;
   Index_type NPNL;
   Index_type NPNR;

   Index_type imin;
   Index_type jmin;
   Index_type kmin;
   Index_type imax;
   Index_type jmax;
   Index_type kmax;

   Index_type jp;
   Index_type kp;
   Index_type nnalls;

   Index_type fpn;
   Index_type lpn;
   Index_type frn;
   Index_type lrn;

   Index_type fpz;
   Index_type lpz;

   Index_type  n_real_zones;
   Index_type  n_real_nodes;
};

std::ostream& operator<<(std::ostream& stream, const ADomain& domain);

//
// Routines for initializing real zone indices for 2d/3d domains.
//
void setRealZones_2d(Index_type* real_zones,
                     const ADomain& domain);

void setRealZones_3d(Index_type* real_zones,
                     const ADomain& domain);

//
// Routines for initializing mesh positions for 2d/3d domains.
//
void setMeshPositions_2d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         const ADomain& domain);

void setMeshPositions_3d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         Real_ptr z, Real_type dz,
                         const ADomain& domain);

} // end namespace apps
} // end namespace rajaperf

#endif  // closing endif for header file include guard
