//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///TODO: REPLACE
/// MC_HISTORY_PARTICLE_TRANSPORT placeholder reference implementation:
///
/// while (tStep < time) {
  //   for (p = 0; p < particles.count; p++) {
  //     std::mt19937_64 RNG(particles.seed[p]);
  //     while(particles.lastEvent[p] != CENSUS && particles.lastEvent[p] != ABSORB && particles.lastEvent[p] != ESCAPE ) {
  //       if (particles.cell[p] == -1)
  //         break;
  //       m = mesh[particles.cell[p]].matID;
  //       std:: cout << "inside cell " << particles.cell[p] <<" with material " << m << "\n";
  //       cross = calcMacroXS(m, particles.group[p]);
  //       scatterDX = calcEventDist(cross.scatter, RNG); 
  //       absDX = calcEventDist(cross.abs, RNG); 
  //       fissionDX = calcEventDist(cross.fission, RNG); 
  //       nufissionDX = calcEventDist(cross.nu_fission, RNG); 
  //       boundaryDX = mesh[particles.cell[p]].getBoundary(particles.pos[p], particles.dir[p], nxt);
  //       particles.calcDX(p); 
  //       minDX = std::min({scatterDX, absDX, fissionDX, nufissionDX, boundaryDX, particles.dx[p]});
  //       particles.pos[p] = {
  //         particles.pos[p][0] + particles.dir[p][0] * minDX,
  //         particles.pos[p][1] + particles.dir[p][1] * minDX,
  //         particles.pos[p][2] + particles.dir[p][2] * minDX
  //       };
  //       particles.dx[p] -= minDX; 
  //       if (particles.E[p] < cutoff || minDX == absDX) {
  //         particles.lastEvent[p] = ABSORB;
  //       else if (minDX == scatterDX) {
  //         particles.lastEvent[p] = SCATTER;
  //         particles.dir[p] = sampleScatter(RNG, particles.E[p]);
  //       }
  //       else if (minDX == fissionDX) {
  //         particles.lastEvent[p] = FISSION;
  //       }
  //       else if (minDX == nufissionDX) {
  //         particles.lastEvent[p] = FISSION;
  //       }
  //       else if (minDX == boundaryDX) {
  //         particles.lastEvent[p] = handleBC(mesh[particles.cell[p]], nxt);
  //         if (particles.lastEvent[p] == BOUNDARY) {
  //           particles.cell[p] = mesh[particles.cell[p]].next[nxt];
  //         }
  //         else if (particles.lastEvent[p] == FISSION || particles.lastEvent[p] == SCATTER) {
  //           particles.dir[p][0] *= -1.0; particles.dir[p][1] *= -1.0; particles.dir[p][2] *= -1.0;
  //         }
  //         else if (particles.lastEvent[p] == ESCAPE) {}
  //       }
  //       else {
  //         particles.lastEvent[p] = CENSUS;
  //         break;
  //       }
  //     }
  //   }
  //   tStep += dT;
  // }
  // CALI_MARK_END("Transport Loop");
///

#include "camp/helpers.hpp"
#include <fstream>
#include <map>
#include <iostream>

#ifndef RAJAPerf_Apps_MC_HISTORY_PARTICLE_TRANSPORT_HPP
#define RAJAPerf_Apps_MC_HISTORY_PARTICLE_TRANSPORT_HPP

#define dT 0.001
#define N_ISOTOPE 68
//#define GROUPS 10
#define cutoff 1e-5


#define MC_HISTORY_PARTICLE_TRANSPORT_DATA_SETUP(vid, partCt) \
  setUp((VariantID)vid, (size_t)partCt);       \
  XS cross = XS();                \
  double scatterDX = 0.0;                       \
  double absDX = 0.0;                           \
  double fissionDX = 0.0;                       \
  double nufissionDX = 0.0;                     \
  double boundaryDX = 0.0;                      \
  double minDX = 0.0;                           \
  u_int32_t nxt = -1;                           \
  int m = -1;                                  \


#define MC_HISTORY_PARTICLE_TRANSPORT_RESET \
  std::copy_n(particles.cell,      particles.count, t_particles.cell); \
  std::copy_n(particles.group,     particles.count, t_particles.group); \
  std::copy_n(particles.lastEvent, particles.count, t_particles.lastEvent); \
  std::copy_n(particles.pos,       particles.count, t_particles.pos); \
  std::copy_n(particles.dir,       particles.count, t_particles.dir); \
  std::copy_n(particles.E,         particles.count, t_particles.E); \
  std::copy_n(particles.dx,        particles.count, t_particles.dx); \
  std::copy_n(particles.seed,      particles.count, t_particles.seed); \
  std::copy_n(particles.prevXS,    particles.count, t_particles.prevXS); \
  std::copy_n(particles.newState,  particles.count, t_particles.newState); \

#define MC_HISTORY_PARTICLE_TRANSPORT_BODY                                                                                                \
  RP_CALI_SUBKERNEL_BEGIN("Transport");     \
  RNG.seed(t_particles.seed[i]);                                                                                             \
  while(t_particles.lastEvent[i] != CENSUS && t_particles.lastEvent[i] != ABSORB && t_particles.lastEvent[i] != ESCAPE ) {  \
    if (t_particles.cell[i] == -1) {                                                                                  \
      std::cout << "LOST particle "<< i << "\n"; break;                                                                                   \
    }                                                                                                                 \
    m = mesh[t_particles.cell[i]].matID;  \
    cross = calcMacroXS(m, t_particles.group[i]);                                                         \
    RP_CALI_SUBKERNEL_BEGIN("Calc Event Distances");                                                                          \
    scatterDX = calcEventDist(cross.scatter, RNG);                                                                    \
    absDX = calcEventDist(cross.abs, RNG);                                                                            \
    fissionDX = calcEventDist(cross.fission, RNG);                                                                    \
    nufissionDX = calcEventDist(cross.nu_fission, RNG);                                                               \
    boundaryDX = mesh[t_particles.cell[i]].getBoundary(t_particles.pos[i], t_particles.dir[i], nxt);                   \
    t_particles.calcDX(i);                                                                                              \
    minDX = std::min({scatterDX, absDX, fissionDX, nufissionDX, boundaryDX, t_particles.dx[i]});                       \
    RP_CALI_SUBKERNEL_END("Calc Event Distances");                                                                            \
    t_particles.pos[i] = {                                                                                              \
      t_particles.pos[i][0] + t_particles.dir[i][0] * minDX,                                                              \
      t_particles.pos[i][1] + t_particles.dir[i][1] * minDX,                                                              \
      t_particles.pos[i][2] + t_particles.dir[i][2] * minDX                                                               \
    };                                                                                                                \
    t_particles.dx[i] -= minDX;                                                                                         \
    if (t_particles.E[i] < cutoff || minDX == absDX) {                                                                  \
      t_particles.lastEvent[i] = ABSORB;                                                                             \
    }\
    else if (minDX == scatterDX) {                                                                                    \
      t_particles.lastEvent[i] = SCATTER;                                                                             \
      t_particles.dir[i] = sampleScatter(RNG, dist, posDist, t_particles.E[i]);                                  \
    }                                                                                                                 \
    else if (minDX == fissionDX) {                                                                                    \
      t_particles.lastEvent[i] = FISSION;                                                                             \
    }                                                                                                                 \
    else if (minDX == nufissionDX) {                                                                                  \
      t_particles.lastEvent[i] = FISSION;                                                                             \
      t_particles.dx[i] *= sampleNuFission(RNG);                                                                      \
    }                                                                                                                 \
    else if (minDX == boundaryDX) {                                                                                   \
      RP_CALI_SUBKERNEL_BEGIN("Handle BCs");                                                                                  \
      t_particles.lastEvent[i] = handleBC(mesh[t_particles.cell[i]], nxt);                                            \
      if (t_particles.lastEvent[i] == BOUNDARY) {                                                                     \
        t_particles.cell[i] = mesh[t_particles.cell[i]].next[nxt];                                                 \
      }                                                                                                               \
      else if (t_particles.lastEvent[i] == FISSION || t_particles.lastEvent[i] == SCATTER) {                          \
        t_particles.dir[i][0] *= -1.0; t_particles.dir[i][1] *= -1.0; t_particles.dir[i][2] *= -1.0;             \
      }                                                                                                               \
      else if (t_particles.lastEvent[i] == ESCAPE) {                                                                  \
      }                                                                                                               \
      RP_CALI_SUBKERNEL_END("Handle BCs");                                                                                  \
    }                                                                                                                 \
    else {                                                                                                            \
      t_particles.lastEvent[i] = CENSUS;                                                                          \
      break;                                                                                                          \
    }                                                                                                                 \
  }                                                                                                                   \
  RP_CALI_SUBKERNEL_END("Transport");                                                                                       \


#include "common/KernelBase.hpp"
#include <random>

namespace rajaperf
{
class RunParams;

namespace apps
{

class MC_HISTORY_PARTICLE_TRANSPORT : public KernelBase
{
public:

  MC_HISTORY_PARTICLE_TRANSPORT(const RunParams& params);

  ~MC_HISTORY_PARTICLE_TRANSPORT();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();

  void runSeqVariant(VariantID vid);

  enum Event {
    CENSUS,
    SCATTER,
    ABSORB,
    BORN,
    FISSION,
    BOUNDARY,
    ESCAPE
  };

  enum BC { // Boundary Conditions
    ELEMENT,
    REFLECT,
    VACUUM,
    SOURCE,
  };

  struct XS { // Cross-sections
    double scatter;
    double abs;
    double fission;
    double nu_fission;
    double total;

    XS();
    XS(std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &posDist);
    
    //inlined operators for shortened code while maintaining performance
    inline XS operator*(const double c) const {
      XS result = *this;
      result.scatter *= c;
      result.abs *= c;
      result.fission *= c;
      result.nu_fission *= c;
      result.total *= c;
      return result;
    }
    inline XS operator+(const XS &sigma) const{
      XS result = *this;
      result.scatter += sigma.scatter;
      result.abs += sigma.abs;
      result.fission += sigma.fission;
      result.nu_fission += sigma.nu_fission;
      result.total += sigma.total;
      return result;
    }
  };

  // Cartesian, Axis-Aligned Mesh
  struct Cell {
    public:
      int ID;
      std::array<double, 6> planes; //-x, +x, -y, +y, -z, +z
      std::array<BC, 6> bc;
      std::array<int, 6> next;
      int matID;

      Cell(); 
      double getBoundary(const std::array<double,3> &pos, const std::array<double,3> &angle, uint32_t &surface_cross);
  };

  // storage of nuclide metadata
  struct Material {
    int isotopeCt;
    Real_ptr conc;
    Int_ptr nucIDs;

    Material();
  };

  // SoA particle storage
  struct Particles {
    public:
      size_t count;
      Int_ptr cell;
      Int_ptr group;       //lower is higher energy
      Event * lastEvent; //doubles as tracker for particle alive
      std::array<double,3> * pos;
      std::array<double,3> * dir;
      Real_ptr E;
      Real_ptr dx;     //remaining distance to census
      uint64_t * seed; 
      XS * prevXS;
      bool * newState; //quick lookup to see if particle state changed and if prevXS valid

      void distribute(const size_t numParticles, std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &dist, std::uniform_real_distribution<double> &posDist, Cell* &mesh, uint64_t dim);
      void calcDX(Index_type i);
  };

  // helpers
  XS calcMacroXS(size_t mat, double E);
  double calcEventDist(double Sigma, std::mt19937_64 &rng);
  double sampleNuFission(std::mt19937_64 &rng);
  Event handleBC(Cell &cell, int face);
  void initMaterials(Material* &mats, VariantID vid);
  void buildMeshCube(size_t side, MC_HISTORY_PARTICLE_TRANSPORT::Cell* &mesh, std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &posDist, VariantID vid);
  static std::array<double, 3> sampleScatter(std::mt19937_64 &rng, std::uniform_real_distribution<double> &dist, std::uniform_real_distribution<double> &posDist, double &E);

// permanent data
private:
  XS * XSTable;
  Material * materials;
  Cell * mesh;
  Particles particles;
  Particles t_particles;
  Real_ptr bins;
  std::uniform_real_distribution<double> dist;
  std::uniform_real_distribution<double> posDist;
  // int GROUPS;
  std::mt19937_64 RNGr;

  bool logging;
  std::mt19937_64 RNG;

  size_t dim;
  size_t GROUPS;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
