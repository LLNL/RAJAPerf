//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "common/RAJAPerfSuite.hpp"
#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Set zone indices for 2d mesh.
//
void setRealZones_2d(Index_type* real_zones,
                     const ADomain& domain)
{
  if (domain.ndims != 2) {
    getCout() << "\n******* ERROR!!! domain is not 2d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;

  Index_type jp = domain.jp;

  Index_type j_stride = (imax - imin);

  for (Index_type j = jmin; j < jmax; j++) {
     for (Index_type i = imin; i < imax; i++) {
        Index_type ip = i + j*jp ;

        Index_type id = (i-imin) + (j-jmin)*j_stride ;
        real_zones[id] = ip;
     }
  }
}

//
// Set zone indices for 3d mesh.
//
void setRealZones_3d(Index_type* real_zones,
                     const ADomain& domain)
{
  if (domain.ndims != 3) {
    getCout() << "\n******* ERROR!!! domain is not 3d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;
  Index_type kmin = domain.kmin;
  Index_type kmax = domain.kmax;

  Index_type jp = domain.jp;
  Index_type kp = domain.kp;

  Index_type j_stride = (imax - imin);
  Index_type k_stride = j_stride * (jmax - jmin);

  for (Index_type k = kmin; k < kmax; k++) {
     for (Index_type j = jmin; j < jmax; j++) {
        for (Index_type i = imin; i < imax; i++) {
           Index_type ip = i + j*jp + k*kp ;

           Index_type id = (i-imin) + (j-jmin)*j_stride + (k-kmin)*k_stride ;
           real_zones[id] = ip;
        }
     }
  }
}

//
// Set mesh positions for 2d mesh.
//
void setMeshPositions_2d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         const ADomain& domain)
{
  if (domain.ndims != 2) {
    getCout() << "\n******* ERROR!!! domain is not 2d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;

  Index_type jp = domain.jp;

  Index_type npnl = domain.NPNL;
  Index_type npnr = domain.NPNR;

  for (Index_type j = jmin - npnl; j < jmax+1 + npnr; j++) {
     for (Index_type i = imin - npnl; i < imax+1 + npnr; i++) {
        Index_type in = i + j*jp ;

        x[in] = i*dx;

        y[in] = j*dy;

     }
  }
}


//
// Set mesh positions for 2d mesh.
//
void setMeshPositions_3d(Real_ptr x, Real_type dx,
                         Real_ptr y, Real_type dy,
                         Real_ptr z, Real_type dz,
                         const ADomain& domain)
{
  if (domain.ndims != 3) {
    getCout() << "\n******* ERROR!!! domain is not 3d *******" << std::endl;
    return;
  }

  Index_type imin = domain.imin;
  Index_type imax = domain.imax;
  Index_type jmin = domain.jmin;
  Index_type jmax = domain.jmax;
  Index_type kmin = domain.kmin;
  Index_type kmax = domain.kmax;

  Index_type jp = domain.jp;
  Index_type kp = domain.kp;

  Index_type npnl = domain.NPNL;
  Index_type npnr = domain.NPNR;

  for (Index_type k = kmin - npnl; k < kmax+1 + npnr; k++) {
     for (Index_type j = jmin - npnl; j < jmax+1 + npnr; j++) {
        for (Index_type i = imin - npnl; i < imax+1 + npnr; i++) {
           Index_type in = i + j*jp + k*kp ;

           x[in] = i*dx;

           y[in] = j*dy;

           z[in] = k*dz;

        }
     }
  }
}

} // end namespace apps
} // end namespace rajaperf
