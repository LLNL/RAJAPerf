//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MC_HISTORY_PARTICLE_TRANSPORT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


MC_HISTORY_PARTICLE_TRANSPORT::MC_HISTORY_PARTICLE_TRANSPORT(const RunParams& params)
  : KernelBase(rajaperf::Apps_MC_HISTORY_PARTICLE_TRANSPORT, params)
{
  dim = params.getMCHistoryCubeSz();
  GROUPS = params.getMCHistoryGroups();
  setDefaultProblemSize(1);
  setDefaultReps(1000);

  setSize(params.getTargetSize(getDefaultProblemSize()),
          params.getReps(getDefaultReps()));

  setChecksumConsistency(ChecksumConsistency::ConsistentPerVariantTuning);
  setChecksumTolerance(ChecksumTolerance::tight);

  setComplexity(Complexity::N);

  setMaxPerfectLoopDimensions(1);
  setProblemDimensionality(1);

  setUsesFeature(Forall);

  addVariantTunings();
}

void MC_HISTORY_PARTICLE_TRANSPORT::setSize(Index_type target_size, Index_type target_reps)
{
  setActualProblemSize( target_size );
  setRunReps( target_reps );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);

  // per particle: cell, group, lastEvent, pos, dir, E, dx, seed, prevXS, newState
  const Index_type part_bytes = 4+4+4+24+24+8+8+8+sizeof(XS)+1;
  // pos, dir, dx, lastEvent, cell, E are read then written by the transport loop
  const Index_type rmw_bytes  = 24+24+8+4+4+8;

  setBytesAllocatedPerRep( 140 * target_size ); // in, out
  setBytesReadPerRep( part_bytes * target_size              // RESET source
                  + dim*dim*dim * sizeof(Cell)           // mesh, read-only
                  + N_ISOTOPE * GROUPS * sizeof(XS)       // XS table, read-only
                  + GROUPS * sizeof(Real_type) );         // bins
  setBytesWrittenPerRep( (part_bytes - rmw_bytes) * target_size );
  setBytesModifyWrittenPerRep( rmw_bytes * target_size );
  setBytesAtomicModifyWrittenPerRep( -1 );
  setFLOPsPerRep( -1 );
}

MC_HISTORY_PARTICLE_TRANSPORT::~MC_HISTORY_PARTICLE_TRANSPORT()
{
}

void MC_HISTORY_PARTICLE_TRANSPORT::setUp(VariantID vid, size_t partCt)
{
  RP_CALI_SUBKERNEL_BEGIN("setUp");

  // Seed distributions and random number generators
  RNGr.seed(18446744073709551557UL);
  dist = std::uniform_real_distribution<double>(-1,1);
  posDist = std::uniform_real_distribution<double>(0,1);
  
  // Construct the flattened Cross-Section table
  RP_CALI_SUBKERNEL_BEGIN("Table setup");
  auto debugXStable = allocDataForInit(XSTable, N_ISOTOPE*GROUPS, vid);
  for (size_t x = 0; x < N_ISOTOPE*GROUPS; x++)
    XSTable[x] = XS(RNGr, posDist);

  RP_CALI_SUBKERNEL_END("Table setup");

  
  // initalize material metadata using copy-pasted values from XSBench's small problem size
  RP_CALI_SUBKERNEL_BEGIN("Init XSBench Materials");
  initMaterials(materials, vid);
  RP_CALI_SUBKERNEL_END("Init XSBench Materials");

  // Initialize dim x dim x dim cube mesh of randomly assigned materials
  RP_CALI_SUBKERNEL_BEGIN("Init Mesh");
  buildMeshCube(dim, mesh, RNGr, posDist, vid);
  RP_CALI_SUBKERNEL_END("Init Mesh");

  // Initialize particle SOA
  RP_CALI_SUBKERNEL_BEGIN("Init Particles");

  allocDataForInit(particles.cell, partCt, vid);
  allocAndInitDataConst(particles.group, partCt, -1, vid);
  allocAndInitDataConst(particles.lastEvent, partCt, BORN, vid);
  allocDataForInit(particles.dir, partCt, vid);
  allocAndInitDataConst(particles.E, partCt, 14.4e6, vid );
  allocDataForInit(particles.dx, partCt, vid);
  allocDataForInit(particles.seed, partCt, vid);
  allocAndInitDataConst(particles.prevXS, partCt, XS(), vid);
  allocAndInitDataConst(particles.newState, partCt, true, vid);
  allocDataForInit(particles.pos, partCt, vid);
  particles.count = partCt;
  particles.distribute(partCt, RNGr, dist, posDist, mesh, dim * dim * dim);

  // initalize scratch SOA to store data for transported particles
  allocDataForInit(t_particles.cell, partCt, vid);
  allocAndInitDataConst(t_particles.group, partCt, -1, vid);
  allocAndInitDataConst(t_particles.lastEvent, partCt, BORN, vid);
  allocDataForInit(t_particles.dir, partCt, vid);
  allocAndInitDataConst(t_particles.E, partCt, 14.4e6, vid );
  allocDataForInit(t_particles.dx, partCt, vid);
  allocDataForInit(t_particles.seed, partCt, vid);
  allocAndInitDataConst(t_particles.prevXS, partCt, XS(), vid);
  allocAndInitDataConst(t_particles.newState, partCt, true, vid);
  allocDataForInit(t_particles.pos, partCt, vid);
  t_particles.count = partCt;

  RP_CALI_SUBKERNEL_END("Init Particles");

  
  // init bins for cross sections
  RP_CALI_SUBKERNEL_BEGIN("Bin Init");
  allocDataForInit(bins, GROUPS, vid);

  double binWidth = 14.1e6/(double)GROUPS, binUpperBound = 0;
  for (size_t i = 0; i < GROUPS; i++) {
    binUpperBound += binWidth;
    bins[i] = binUpperBound;
  }
  RP_CALI_SUBKERNEL_END("Bin Init");

  RP_CALI_SUBKERNEL_END("setUp");

}

void MC_HISTORY_PARTICLE_TRANSPORT::initMaterials(Material* &mats, VariantID vid) {
  size_t matCt = 12;
  auto debugMeshInit = allocDataForInit(mats, matCt, vid);

  // metadata copied from XSBench, 68 nuclides
  mats[0].isotopeCt = 34;
  mats[1].isotopeCt  = 5;
	mats[2].isotopeCt  = 4;
	mats[3].isotopeCt  = 4;
	mats[4].isotopeCt  = 27;
	mats[5].isotopeCt  = 21;
	mats[6].isotopeCt  = 21;
	mats[7].isotopeCt  = 21;
	mats[8].isotopeCt  = 21;
	mats[9].isotopeCt  = 21;
	mats[10].isotopeCt = 9;
	mats[11].isotopeCt = 9;
  
  const int ids0[] = { 58, 59, 60, 61, 40, 42, 43, 44, 45, 46, 1, 2, 3, 7,
	                 8, 9, 10, 29, 57, 47, 48, 0, 62, 15, 33, 34, 52, 53, 
	                 54, 55, 56, 18, 23, 41 };
  const int ids1[] = { 63, 64, 65, 66, 67 };
  const int ids2[] = { 24, 41, 4, 5 };
  const int ids3[] = { 24, 41, 4, 5 };
  const int ids4[] = { 19, 20, 21, 22, 35, 36, 37, 38, 39, 25, 27, 28, 29,
	                 30, 31, 32, 26, 49, 50, 51, 11, 12, 13, 14, 6, 16,
	                 17 }; 
  const int ids5[] = { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
  const int ids6[] = { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
  const int ids7[] = { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
  const int ids8[] = { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
  const int ids9[] = { 24, 41, 4, 5, 19, 20, 21, 22, 35, 36, 37, 38, 39, 25,
	                 49, 50, 51, 11, 12, 13, 14 };
  const int ids10[] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 };
  const int ids11[] = { 24, 41, 4, 5, 63, 64, 65, 66, 67 }; 

  for (int i = 0; i < matCt; i++) {
    allocDataForInit(mats[i].nucIDs, mats[i].isotopeCt, vid); 
  }

  std::copy(std::begin(ids0), std::end(ids0), mats[0].nucIDs);
  std::copy(std::begin(ids1), std::end(ids1), mats[1].nucIDs);
  std::copy(std::begin(ids2), std::end(ids2), mats[2].nucIDs);
  std::copy(std::begin(ids3), std::end(ids3), mats[3].nucIDs);
  std::copy(std::begin(ids4), std::end(ids4), mats[4].nucIDs);
  std::copy(std::begin(ids5), std::end(ids5), mats[5].nucIDs);
  std::copy(std::begin(ids6), std::end(ids6), mats[6].nucIDs);
  std::copy(std::begin(ids7), std::end(ids7), mats[7].nucIDs);
  std::copy(std::begin(ids8), std::end(ids8), mats[8].nucIDs);
  std::copy(std::begin(ids9), std::end(ids9), mats[9].nucIDs);
  std::copy(std::begin(ids10), std::end(ids10), mats[10].nucIDs);
  std::copy(std::begin(ids11), std::end(ids11), mats[11].nucIDs);


  // random assignement of materials ot mesh cells
  for (int i = 0; i < matCt; i++) {
    auto debugConcs = allocDataForInit(mats[i].conc, mats[i].isotopeCt, vid);
    for (int m = 0; m < materials[i].isotopeCt; m++) {
      materials[i].conc[m] = 20 * posDist(RNGr);
    }
  }
  
}

void MC_HISTORY_PARTICLE_TRANSPORT::buildMeshCube(size_t side, Cell* &mesh, std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &posDist, VariantID vid) {
  if (side == 0) {
    throw std::invalid_argument("buildMeshCube side must be nonzero");
  }

  auto debugBuildMesh = allocDataForInit(mesh, side * side * side, vid);

  constexpr double cellWidth = 0.01;
  size_t layerSize = side * side;
  size_t cubeSz = side * side * side;

  auto index = [side, layerSize](size_t row, size_t col, size_t layer) {
    return static_cast<int>(layer * layerSize + col * side + row);
  };

  const double halfWidth = 0.5 * side * cellWidth;
  
  for (size_t layer = 0; layer < side; layer++) {
    for (size_t col = 0; col < side; col++) {
      for (size_t row = 0; row < side; row++) {
        int i = index(row, col, layer);
        MC_HISTORY_PARTICLE_TRANSPORT::Cell &cell = mesh[i];
        cell.ID = i;
        cell.next = {
          row > 0 ? index(row - 1, col, layer) : -1,
          row + 1 < side ? index(row + 1, col, layer) : -1,
          col + 1 < side ? index(row, col + 1, layer) : -1,
          col > 0 ? index(row, col - 1, layer) : -1,
          layer > 0 ? index(row, col, layer - 1) : -1,
          layer + 1 < side ? index(row, col, layer + 1) : -1
        };

        if (row == 0) cell.bc[0] = REFLECT;
        if (row + 1 == side) cell.bc[1] = REFLECT;
        if (col + 1 == side) cell.bc[2] = REFLECT;
        if (col == 0) cell.bc[3] = REFLECT;
        if (layer == 0) cell.bc[4] = VACUUM;
        if (layer + 1 == side) cell.bc[5] = VACUUM;

        double xMin = -halfWidth + row * cellWidth;
        double yMax = halfWidth - col * cellWidth;
        double zMin = -halfWidth + layer * cellWidth;
        cell.planes = {
          xMin, xMin + cellWidth,
          yMax - cellWidth, yMax,
          zMin, zMin + cellWidth
        };

        cell.matID = 11 * posDist(RNGr);      
      }
    }
  }
}

void MC_HISTORY_PARTICLE_TRANSPORT::updateChecksum(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  addToChecksum(0);
}

void MC_HISTORY_PARTICLE_TRANSPORT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  deallocData(XSTable, vid);

  for (size_t m = 0; m < 12; m++) {
    deallocData(materials[m].conc , vid);
    deallocData(materials[m].nucIDs, vid);
  }
  deallocData(materials, vid);

  deallocData(mesh, vid);
  deallocData(bins, vid);

  //particles 
  deallocData(particles.cell, vid);
  deallocData(particles.dir, vid);
  deallocData(particles.dx, vid);
  deallocData(particles.E, vid);
  deallocData(particles.group, vid);
  deallocData(particles.lastEvent, vid);
  deallocData(particles.newState, vid);
  deallocData(particles.pos, vid);
  deallocData(particles.prevXS, vid);
  deallocData(particles.seed, vid);

  deallocData(t_particles.cell, vid);
  deallocData(t_particles.dir, vid);
  deallocData(t_particles.dx, vid);
  deallocData(t_particles.E, vid);
  deallocData(t_particles.group, vid);
  deallocData(t_particles.lastEvent, vid);
  deallocData(t_particles.newState, vid);
  deallocData(t_particles.pos, vid);
  deallocData(t_particles.prevXS, vid);
  deallocData(t_particles.seed, vid);
}


MC_HISTORY_PARTICLE_TRANSPORT::XS::XS() {
  scatter = 0;
  abs = 0;
  fission = 0;
  nu_fission = 0;
  total = 0;
}

MC_HISTORY_PARTICLE_TRANSPORT::XS::XS(std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &posDist) {
  scatter = posDist(RNGr);
  abs = posDist(RNGr);
  fission = posDist(RNGr);
  nu_fission = posDist(RNGr);
  total = scatter + abs + fission + nu_fission;
}

// calculate aggregate cross section for a material
MC_HISTORY_PARTICLE_TRANSPORT::XS MC_HISTORY_PARTICLE_TRANSPORT::calcMacroXS(size_t mat, double E) {
  RP_CALI_SUBKERNEL_BEGIN("calcMacroXS");
  XS Sigma;
  size_t bin = std::upper_bound(bins, bins + GROUPS, E) - bins;
  for (int i = 0; i < materials[mat].isotopeCt; i++) {
    Sigma = Sigma + XSTable[GROUPS * materials[mat].nucIDs[i] + bin] * materials[mat].conc[i];
  }
  RP_CALI_SUBKERNEL_END("calcMacroXS");
  return Sigma;
}

// sample MFPs
double MC_HISTORY_PARTICLE_TRANSPORT::calcEventDist(double Sigma, std::mt19937_64 &rng) {
  return posDist(rng)/Sigma;
}

// sample direction and loss of energy during a scatter event
std::array<double, 3> MC_HISTORY_PARTICLE_TRANSPORT::sampleScatter(std::mt19937_64 &rng, std::uniform_real_distribution<double> &dist, std::uniform_real_distribution<double> &posDist, double &E) {
  double x = dist(rng), y = dist(rng), z = dist(rng);  
  double norm = std::sqrt(x*x + y*y + z*z);
  x /= norm; y/= norm; z/= norm; 

  E *= posDist(rng);

  return {x,y,z};
}

// sample extended lifetime for fissioned particle to simulate fission products
double MC_HISTORY_PARTICLE_TRANSPORT::sampleNuFission(std::mt19937_64 &rng) {
  return 5 * (dist(rng) + 1);
}


MC_HISTORY_PARTICLE_TRANSPORT::Cell::Cell() {
    for (int i = 0; i < 6; i++){
      next[i] = -1;
      bc[i] = ELEMENT;
      planes[i] = (i % 2 == 0) ? -1.0 : 1.0;
    }
    ID = -1;
    matID = 0;
  }

// calculate distance to boundary particle traveling toward from current direction and position
double MC_HISTORY_PARTICLE_TRANSPORT::Cell::getBoundary(const std::array<double,3> &pos, const std::array<double,3> &angle, uint32_t &surface_cross) {
  double min_dist = std::numeric_limits<double>::infinity();
  double dist = 0.0;
  uint32_t index;
  // only check the positive or negative surface
  for (uint32_t i = 0; i < 3; i++) {
    if (std::abs(angle[i]) < 1.0e-14) {
      continue;
    }
    index = 2 * i + ( (angle[i] > 0) ? 1 : 0 ); 
    dist = (planes[index] - pos[i]) / angle[i]; 
    if (dist >= 0.0 && dist < min_dist) {
      min_dist = dist;
      surface_cross = index;
    }
  }
  return min_dist;
}

// state handler for boundary conditions
MC_HISTORY_PARTICLE_TRANSPORT::Event MC_HISTORY_PARTICLE_TRANSPORT::handleBC(Cell &cell, int face) {
  Event pState;
  switch (cell.bc[face]) {
  case ELEMENT:
    pState = BOUNDARY;
    break;
  case REFLECT:
    pState = SCATTER;
    break;
  case VACUUM:
    pState = ESCAPE;
    break;
  case SOURCE:
    pState = FISSION;
    break;
  }
  return pState;
}

// distribute particles uniformly across mesh
void MC_HISTORY_PARTICLE_TRANSPORT::Particles::distribute(const size_t numParticles, std::mt19937_64 &RNGr, std::uniform_real_distribution<double> &dist, std::uniform_real_distribution<double> &posDist, Cell* &mesh, uint64_t dim) {

  std::uniform_int_distribution<int> cellDist(0, dim - 1);

  for (size_t i = 0; i < numParticles; i++) {
    seed[i] = i*3;
    dir[i] = sampleScatter(RNGr, dist, posDist, E[i]);
    calcDX(i);

    cell[i] = cellDist(RNGr);
    const Cell &c = mesh[cell[i]];
    pos[i] = {c.planes[0] + 0.001 * posDist(RNGr), 
              c.planes[2] + 0.001 * posDist(RNGr),
              c.planes[4] + 0.001 * posDist(RNGr)};
  }
}

// convert MeV to m, assuming neutron
void MC_HISTORY_PARTICLE_TRANSPORT::Particles::calcDX(Index_type i) {
  dx[i] = sqrt((2/1.7e-27) * (E[i]/(double)6.242e12) ) * dT;
}

} // end namespace apps
} // end namespace rajaperf
