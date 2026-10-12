//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FEMSWEEP kernel reference implementation:
///
/// for (Index_type ag = 0; ag < na * ng; ++ag)
/// {
///   const Index_type a = ag / ng;
///   const Index_type g = ag % ng;
///   // number and offset of hyperplanes for this angle
///   const Index_type nhp = nhpaa_r[a];
///   const Index_type ohp = ohpaa_r[a];
///   // elements in this hyperplanes processed so far
///   Index_type s_nehp_done = 0;
///   Real_type A[ND*ND];
//    Real_type b[ND];
///   // This factor helps maintain stability in the solution of the matrix solve
///   // by eliminating the perturbation of the right-hand side.
///   Real_type Ffactor = fmax(sin(Adat[order_r[a*ne]*ND*ND + a*ne*ND*ND]) - 2.0, 0.0);
///   for (Index_type hp = 0; hp < nhp; ++hp) // loop over hyperplanes
///   {
///      // number of element in this hyperplane
///      const Index_type nehp = phpaa_r[ohp + hp];
///      // loop over all elements of this hyperplane
///      for (Index_type k = 0; k < nehp; ++k)
///      {
///         const Index_type e = order_r[k + s_nehp_done + a * ne];
///         for (Index_type j = 0; j < ND; ++j) // load B & A
///         {
///            b[j] = Bdat[j + e * ND + a * ne * ND];
///            for (Index_type i = 0; i < ND; ++i)
///            {
///               A[i + j * ND] = Adat[i + j * ND + e * ND * ND + a * ne * ND * ND];
///            }
///         }
///         for (Index_type face = 0; face < NLF; ++face) // local faces
///         {
///            const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]];
///            const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl; // signed face
///            const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1; // side
///            if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) // non-reentrant and outgoing, or boundary
///            {
///               continue;
///            }
///            for (Index_type j = 0; j < FDS; ++j) // face dofs
///            {
///               const Index_type ffj = f * FDS + j;
///               const Index_type djs = s == 0 ? idx1[ffj] : idx2[ffj];
///               Real_type F = 0.0;
///               for (Index_type i = 0; i < FDS; i++) // face dofs
///               {
///                  const Index_type ffi = f * FDS + i;
///                  const Index_type dis = s == 0 ? idx2[ffi] : idx1[ffi];
///                  // Fdat : 4 x 4 x 2 x nf_int x na
///                  // Note: s ^ 1 == 1 if s == 0, and s ^ 1 == 0 if s == 1.
///                  // The solution remains bounded when matrices and indirection arrays from a
///                  // finite element discretrization are used, but randomly generating matrix 
///                  // data can result in exponential growth instead. A specific trick of 
///                  // multiplying by zero, without the compiler optimizing it away, is 
///                  // necessary to prevent unbounded growth and avoid computing NaNs.
///                  F += Ffactor * Fdat[i + j * FDS + (s ^ 1) * FDS * FDS + f * 2 * FDS * FDS + a * sharedinteriorfaces * 2 * FDS * FDS] * Xdat[dis + g * ND * ne + a * ng * ND * ne];
///               } // i
///               b[djs % ND] -= F;
///            } // j
///         }    // local faces
///         const Real_type s = Sgdat[e + g * ne];
///         // A = A + s * M0, b is the result
///         SolveLinearSystemNxN<ND>(A, s, &M0dat[0 + 0 * ND + e * ND * ND], b, &Xdat[e * ND + g * ND * ne + a * ng * ND * ne]);  // 8x8 solve
///      }  // thread loop elems in hp
///      s_nehp_done += nehp;
///   } // hyperplanes
/// }
///
/// The RAJA variants of this kernel use RAJA multi-dimensional data layouts
/// and views to do the same thing without explicit index calculations (see
/// the loop body definitions below).
///

#ifndef RAJAPerf_Apps_FEMSWEEP_HPP
#define RAJAPerf_Apps_FEMSWEEP_HPP

#include "common/KernelBase.hpp"

#include "RAJA/RAJA.hpp"

#include "FEMSWEEPMeshGen.hpp"

constexpr RAJA::Index_type ND = 8;   // number of corners per element
constexpr RAJA::Index_type NLF = 6;  // number of faces per element
constexpr RAJA::Index_type FDS = 4;  // number of DOFs per face

#define FEMSWEEP_UNROLL_TUNING_NAME "unroll"

#define FEMSWEEP_DATA_SETUP \
  Real_ptr Bdat = m_Bdat; \
  Real_ptr Adat = m_Adat; \
  Real_ptr Fdat = m_Fdat; \
  Real_ptr Xdat = m_Xdat; \
  Real_ptr Sgdat = m_Sgdat; \
  Real_ptr M0dat = m_M0dat; \
\
  Index_type ne = m_ne; \
  Index_type na = m_na; \
  Index_type ng = m_ng; \
  Index_type sharedinteriorfaces = m_sharedinteriorfaces; \
\
  Index_ptr nhpaa_r = m_nhpaa_r;  \
  Index_ptr ohpaa_r = m_ohpaa_r;  \
  Index_ptr phpaa_r = m_phpaa_r;  \
  Index_ptr order_r = m_order_r;  \
\
  Index_ptr AngleElem2FaceType = m_AngleElem2FaceType; \
  Index_ptr elem_to_faces      = m_elem_to_faces     ; \
  Index_ptr F_g2l              = m_F_g2l             ; \
  Index_ptr idx1               = m_idx1              ; \
  Index_ptr idx2               = m_idx2              ; \


#define FEMSWEEP_KERNEL_SETUP \
  const Index_type nhp = nhpaa_r[a]; \
  const Index_type ohp = ohpaa_r[a]; \
  Real_type Ffactor = fmax(sin(Adat[order_r[a*ne]*ND*ND + a*ne*ND*ND]) - 2.0, 0.0);

namespace rajaperf
{

#define FEMSWEEP_ROLL
#define FEMSWEEP_UNROLL RAJA_UNROLL

// LU factorization with no pivoting
#define FEMSWEEP_DEFINE_SOLVE_LINEAR_SYSTEM_NXN(FUNCTION_NAME, UNROLL_MODE) \
template <Index_type N> \
RAJA_HOST_DEVICE RAJA_INLINE \
void FUNCTION_NAME(Real_ptr        A, \
                   const Real_type s, \
                   Real_const_ptr  M, \
                   Real_ptr        b, \
                   Real_ptr        x) \
{ \
  Real_type tempA[N][N]; \
  Real_type L[N][N]; \
  Real_type U[N][N]; \
  Real_type D[N]; \
  Real_type tempx[N]; \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = 0; ii < N; ++ii) \
  { \
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type jj = 0; jj < N; ++jj) \
    { \
      tempA[ii][jj] = A[ii * N + jj] + s * M[ii * N + jj]; \
      L[ii][jj] = 0.0; \
      if ( ii == jj ) \
      { \
        U[ii][jj] = 1.0; \
      } \
      else \
      { \
        U[ii][jj] = 0.0; \
      } \
    } \
  } \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = 0; ii < N; ++ii) \
  { \
    L[ii][0] = tempA[ii][0]; \
  } \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = 1; ii < N; ++ii) \
  { \
    U[0][ii] = tempA[0][ii]/tempA[0][0]; \
  } \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = 1; ii < N; ++ii) \
  { \
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type jj = ii; jj < N; ++jj) \
    { \
      Real_type sum = 0.0; \
      FEMSWEEP_##UNROLL_MODE \
      for (Index_type kk = 0; kk < jj; ++kk) \
      { \
        sum += L[jj][kk] * U[kk][ii]; \
      } \
      L[jj][ii] = tempA[jj][ii] - sum; \
    } \
\
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type jj = ii + 1; jj < N; ++jj) \
    { \
      Real_type sum = 0.0; \
      FEMSWEEP_##UNROLL_MODE \
      for (Index_type kk = 0; kk < ii; ++kk) \
      { \
        sum += L[ii][kk] * U[kk][jj]; \
      } \
      U[ii][jj] = (tempA[ii][jj] - sum)/L[ii][ii]; \
    } \
  } \
\
  D[0] = b[0]/L[0][0]; \
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = 1; ii < N; ++ii) \
  { \
    Real_type sum = 0.0; \
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type jj = 0; jj < ii; ++jj) \
    { \
      sum += L[ii][jj] * D[jj]; \
    } \
    D[ii] = (b[ii] - sum)/L[ii][ii]; \
  } \
\
  x[N-1] = tempx[N-1] = D[N-1]; \
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type ii = N - 1 - 1; ii > -1; --ii) \
  { \
    Real_type sum = 0.0; \
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type jj = ii + 1; jj < N; ++jj) \
    { \
      sum += U[ii][jj] * tempx[jj]; \
    } \
    x[ii] = tempx[ii] = D[ii] - sum; \
  } \
}

FEMSWEEP_DEFINE_SOLVE_LINEAR_SYSTEM_NXN(SolveLinearSystemNxN, ROLL)
FEMSWEEP_DEFINE_SOLVE_LINEAR_SYSTEM_NXN(SolveLinearSystemNxNUnroll, UNROLL)

#undef FEMSWEEP_DEFINE_SOLVE_LINEAR_SYSTEM_NXN

#define FEMSWEEP_DEFINE_HYPERPLANE_ELEMENT(FUNCTION_NAME, SOLVE, UNROLL_MODE) \
RAJA_HOST_DEVICE RAJA_INLINE \
void FUNCTION_NAME(Real_const_ptr Bdat, \
                   Real_const_ptr Adat, \
                   Real_const_ptr Fdat, \
                   Real_ptr       Xdat, \
                   Real_const_ptr Sgdat, \
                   Real_const_ptr M0dat, \
                   const Index_type ne, \
                   const Index_type ng, \
                   const Index_type sharedinteriorfaces, \
                   Index_type const * order_r, \
                   Index_type const * AngleElem2FaceType, \
                   Index_type const * elem_to_faces, \
                   Index_type const * F_g2l, \
                   Index_type const * idx1, \
                   Index_type const * idx2, \
                   const Index_type a, \
                   const Index_type g, \
                   const Index_type k, \
                   const Index_type nehp_pos, \
                   const Real_type Ffactor) \
{ \
  Real_type A[ND*ND]; \
  Real_type b[ND]; \
  const Index_type e = order_r[k + nehp_pos + a * ne]; \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type j = 0; j < ND; ++j) \
  { \
    b[j] = Bdat[j + e * ND + a * ne * ND]; \
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type i = 0; i < ND; ++i) \
    { \
      A[i + j * ND] = Adat[i + j * ND + e * ND * ND + a * ne * ND * ND]; \
    } \
  } \
\
  FEMSWEEP_##UNROLL_MODE \
  for (Index_type face = 0; face < NLF; ++face) \
  { \
    const Index_type sf_gl = F_g2l[elem_to_faces[NLF * e + face]]; \
    const Index_type f = sf_gl >= 0 ? sf_gl : -1 - sf_gl; \
    const Index_type s = (e == idx1[f * FDS] / ND) ? 0 : 1; \
    if ((AngleElem2FaceType[face + e * NLF + a * ne * NLF] == 0) || (sf_gl < 0)) \
    { \
      continue; \
    } \
\
    FEMSWEEP_##UNROLL_MODE \
    for (Index_type j = 0; j < FDS; ++j) \
    { \
      const Index_type ffj = f * FDS + j; \
      const Index_type djs = s == 0 ? idx1[ffj] : idx2[ffj]; \
      Real_type F = 0.0; \
      FEMSWEEP_##UNROLL_MODE \
      for (Index_type i = 0; i < FDS; ++i) \
      { \
        const Index_type ffi = f * FDS + i; \
        const Index_type dis = s == 0 ? idx2[ffi] : idx1[ffi]; \
        F += Ffactor * \
             Fdat[i + j * FDS + (s ^ 1) * FDS * FDS + \
                  f * 2 * FDS * FDS + \
                  a * sharedinteriorfaces * 2 * FDS * FDS] * \
             Xdat[dis + g * ND * ne + a * ng * ND * ne]; \
      } \
      b[djs % ND] -= F; \
    } \
  } \
\
  const Real_type s = Sgdat[e + g * ne]; \
  SOLVE<ND>(A, s, &M0dat[0 + 0 * ND + e * ND * ND], b, &Xdat[e * ND + g * ND * ne + a * ng * ND * ne]); \
}

FEMSWEEP_DEFINE_HYPERPLANE_ELEMENT(
    femsweepHyperplaneElementRolled, SolveLinearSystemNxN, ROLL)
FEMSWEEP_DEFINE_HYPERPLANE_ELEMENT(
    femsweepHyperplaneElementUnrolled, SolveLinearSystemNxNUnroll, UNROLL)

#undef FEMSWEEP_DEFINE_HYPERPLANE_ELEMENT
#undef FEMSWEEP_UNROLL
#undef FEMSWEEP_ROLL

template <bool unroll>
RAJA_HOST_DEVICE RAJA_INLINE
void femsweepHyperplaneElement(Real_const_ptr Bdat,
                               Real_const_ptr Adat,
                               Real_const_ptr Fdat,
                               Real_ptr Xdat,
                               Real_const_ptr Sgdat,
                               Real_const_ptr M0dat,
                               const Index_type ne,
                               const Index_type ng,
                               const Index_type sharedinteriorfaces,
                               Index_type const * order_r,
                               Index_type const * AngleElem2FaceType,
                               Index_type const * elem_to_faces,
                               Index_type const * F_g2l,
                               Index_type const * idx1,
                               Index_type const * idx2,
                               const Index_type a,
                               const Index_type g,
                               const Index_type k,
                               const Index_type nehp_pos,
                               const Real_type Ffactor)
{
  if constexpr (unroll)
  {
    femsweepHyperplaneElementUnrolled(Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
                                      ne, ng, sharedinteriorfaces, order_r,
                                      AngleElem2FaceType, elem_to_faces, F_g2l,
                                      idx1, idx2, a, g, k, nehp_pos, Ffactor);
  }
  else
  {
    femsweepHyperplaneElementRolled(Bdat, Adat, Fdat, Xdat, Sgdat, M0dat,
                                    ne, ng, sharedinteriorfaces, order_r,
                                    AngleElem2FaceType, elem_to_faces, F_g2l,
                                    idx1, idx2, a, g, k, nehp_pos, Ffactor);
  }
}


class RunParams;

namespace apps
{

class FEMSWEEP : public KernelBase
{
public:

  FEMSWEEP(const RunParams& params);

  ~FEMSWEEP();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();

  template < bool unroll >
  void runSeqVariant(VariantID vid);
  template < bool unroll >
  void runOpenMPVariant(VariantID vid);

  template < size_t block_size, bool unroll, size_t min_blocks_per_sm >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size, bool unroll >
  void runHipVariantImpl(VariantID vid);

private:
#if defined(RAJA_ENABLE_HIP)
  static const size_t default_gpu_block_size = 64;
#else
  static const size_t default_gpu_block_size = 64;
#endif
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                                         integer::MultipleOf<32>>;

  AngularQuadratureLite * m_angularquadrature;
  MeshGenerator * m_meshgen;

  Index_type m_nx;
  Index_type m_ny;
  Index_type m_nz;
  Index_type m_ne;
  Index_type m_na;
  Index_type m_ng;

  Index_type m_sharedinteriorfaces;
  Index_type m_boundaryfaces;
  Index_type m_hplanes;

  Real_ptr m_Bdat;
  Real_ptr m_Adat;
  Real_ptr m_Fdat;
  Real_ptr m_Sgdat;
  Real_ptr m_M0dat;
  Real_ptr m_Xdat;

  Index_type m_Blen;
  Index_type m_Alen;
  Index_type m_Flen;
  Index_type m_Sglen;
  Index_type m_M0len;
  Index_type m_Xlen;

  Index_type m_nhpaa_rlen;
  Index_type m_ohpaa_rlen;
  Index_type m_phpaa_rlen;
  Index_type m_order_rlen;

  Index_type m_AngleElem2FaceTypelen;
  Index_type m_elem_to_faceslen;
  Index_type m_F_g2llen;
  Index_type m_idx1len;
  Index_type m_idx2len;

  // Mesh data
  Index_ptr m_nhpaa_r;
  Index_ptr m_ohpaa_r;
  Index_ptr m_phpaa_r;
  Index_ptr m_order_r;

  Index_ptr m_AngleElem2FaceType;
  Index_ptr m_elem_to_faces     ;
  Index_ptr m_F_g2l             ;
  Index_ptr m_idx1              ;
  Index_ptr m_idx2              ;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
