//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_AppsData_HPP
#define RAJAPerf_AppsData_HPP

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

   ADomain( Index_type rzmax, Index_type ndims ) 
      : ndims(ndims), NPNL(2), NPNR(1)
   {
      imin = NPNL;
      jmin = NPNL;
      imax = rzmax + NPNR;
      jmax = rzmax + NPNR;
      jp = imax - imin + 1 + NPNL + NPNR;
      n_real_zones = (imax - imin);
      n_real_nodes = (imax+1 - imin);

      if ( ndims == 2 ) {
         kmin = 0;
         kmax = 0;
         kp = 0;
         nnalls = jp * (jmax - jmin + 1 + NPNL + NPNR) ;
         n_real_zones *= (jmax - jmin);
         n_real_nodes *= (jmax+1 - jmin);
      } else if ( ndims == 3 ) {
         kmin = NPNL;
         kmax = rzmax + NPNR;
         kp = jp * (jmax - jmin + 1 + NPNL + NPNR);
         nnalls = kp * (kmax - kmin + 1 + NPNL + NPNR) ;
         n_real_zones *= (jmax - jmin) * (kmax - kmin);
         n_real_nodes *= (jmax+1 - jmin) * (kmax+1 - kmin);
      }

      fpn = 0;
      lpn = nnalls - 1;
      frn = fpn + NPNL * (kp + jp) + NPNL;
      lrn = lpn - NPNR * (kp + jp) - NPNR;

      fpz = frn - jp - kp - 1;
      lpz = lrn;
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
