
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SW4CK_KERNEL_5 kernel reference implementation:
/// https://github.com/LLNL/SW4CK
///
///   for (int k = kstart; k <= klast - 2; k++)
///     for (int j = jfirst + 2; j <= jlast - 2; j++)
///       for (int i = ifirst + 2; i <= ilast - 2; i++) {
///
///           // 5 ops
///           float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
///           float_sw4 istry = 1 / (stry(j));
///           float_sw4 istrx = 1 / (strx(i));
///           float_sw4 istrxy = istry * istrx;
///
///           float_sw4 r1 = 0, r2 = 0, r3 = 0;
///
///           // pp derivative (u) (u-eq)
///           // 53 ops, tot=58
///           float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
///                            met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                            strx(i - 2);
///           float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
///                            met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                            strx(i - 1);
///           float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
///                            met(1, i, j, k) * strx(i);
///           float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
///                            met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                            strx(i + 1);
///           float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
///                            met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                            strx(i + 2);
///
///           float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
///           float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           float_sw4 mux4 = cof4 - tf * (cof3 + cof5);
///
///           r1 = r1 + i6 *
///                         (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
///                          mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
///                          mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
///                          mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
///                         istry;
///
///           // qq derivative (u) (u-eq)
///           // 43 ops, tot=101
///           cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
///                  stry(j - 2);
///           cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
///                  stry(j - 1);
///           cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
///           cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
///                  stry(j + 1);
///           cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
///                  stry(j + 2);
///
///           mux1 = cof2 - tf * (cof3 + cof1);
///           mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           mux4 = cof4 - tf * (cof3 + cof5);
///
///           r1 = r1 + i6 *
///                         (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
///                          mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
///                          mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
///                          mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
///                         istrx;
///
///           // pp derivative (v) (v-eq)
///           // 43 ops, tot=144
///           cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                  strx(i - 2);
///           cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                  strx(i - 1);
///           cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
///           cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                  strx(i + 1);
///           cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                  strx(i + 2);
///
///           mux1 = cof2 - tf * (cof3 + cof1);
///           mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           mux4 = cof4 - tf * (cof3 + cof5);
///
///           r2 = r2 + i6 *
///                         (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +
///                          mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +
///                          mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +
///                          mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *
///                         istry;
///
///           // qq derivative (v) (v-eq)
///           // 53 ops, tot=197
///           cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) * met(1, i, j - 2, k) *
///                  met(1, i, j - 2, k) * stry(j - 2);
///           cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) * met(1, i, j - 1, k) *
///                  met(1, i, j - 1, k) * stry(j - 1);
///           cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
///                  met(1, i, j, k) * stry(j);
///           cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) * met(1, i, j + 1, k) *
///                  met(1, i, j + 1, k) * stry(j + 1);
///           cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) * met(1, i, j + 2, k) *
///                  met(1, i, j + 2, k) * stry(j + 2);
///           mux1 = cof2 - tf * (cof3 + cof1);
///           mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           mux4 = cof4 - tf * (cof3 + cof5);
///
///           r2 = r2 + i6 *
///                         (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +
///                          mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +
///                          mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +
///                          mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *
///                         istrx;
///
///           // pp derivative (w) (w-eq)
///           // 43 ops, tot=240
///           cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                  strx(i - 2);
///           cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                  strx(i - 1);
///           cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);
///           cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                  strx(i + 1);
///           cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                  strx(i + 2);
///
///           mux1 = cof2 - tf * (cof3 + cof1);
///           mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           mux4 = cof4 - tf * (cof3 + cof5);
///
///           r3 = r3 + i6 *
///                         (mux1 * (u(3, i - 2, j, k) - u(3, i, j, k)) +
///                          mux2 * (u(3, i - 1, j, k) - u(3, i, j, k)) +
///                          mux3 * (u(3, i + 1, j, k) - u(3, i, j, k)) +
///                          mux4 * (u(3, i + 2, j, k) - u(3, i, j, k))) *
///                         istry;
///
///           // qq derivative (w) (w-eq)
///           // 43 ops, tot=283
///           cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
///                  stry(j - 2);
///           cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
///                  stry(j - 1);
///           cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
///           cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
///                 stry(j + 1);
///           cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
///                  stry(j + 2);
///           mux1 = cof2 - tf * (cof3 + cof1);
///           mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           mux4 = cof4 - tf * (cof3 + cof5);
///
///           r3 = r3 + i6 *
///                         (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +
///                          mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +
///                          mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +
///                          mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *
///                         istrx;
///
///           // All rr-derivatives at once
///           // averaging the coefficient
///           // 54*8*8+25*8 = 3656 ops, tot=3939
///           float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;
///           //#pragma unroll 8
/// #ifdef MAGIC_SYNC
/// 	    __syncthreads();
/// #endif
///           for (int q = nk - 7; q <= nk; q++) {
///             mucofu2 = 0;
///             mucofuv = 0;
///             mucofuw = 0;
///             mucofvw = 0;
///             mucofv2 = 0;
///             mucofw2 = 0;
/// #ifdef AMD_UNROLL_FIX
/// #pragma unroll 8
/// #endif
///             for (int m = nk - 7; m <= nk; m++) {
///               mucofu2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                          ((2 * mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
///                               strx(i) * met(2, i, j, m) * strx(i) +
///                           mu(i, j, m) * (met(3, i, j, m) * stry(j) *
///                                              met(3, i, j, m) * stry(j) +
///                                          met(4, i, j, m) * met(4, i, j, m)));
///               mucofv2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                          ((2 * mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
///                               stry(j) * met(3, i, j, m) * stry(j) +
///                           mu(i, j, m) * (met(2, i, j, m) * strx(i) *
///                                              met(2, i, j, m) * strx(i) +
///                                          met(4, i, j, m) * met(4, i, j, m)));
///               mucofw2 +=
///                   acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                   ((2 * mu(i, j, m) + la(i, j, m)) * met(4, i, j, m) *
///                        met(4, i, j, m) +
///                    mu(i, j, m) *
///                        (met(2, i, j, m) * strx(i) * met(2, i, j, m) * strx(i) +
///                         met(3, i, j, m) * stry(j) * met(3, i, j, m) * stry(j)));
///               mucofuv += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                          (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
///                          met(3, i, j, m);
///               mucofuw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                          (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *
///                          met(4, i, j, m);
///               mucofvw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *
///                          (mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *
///                          met(4, i, j, m);
///             }
///
///             // Computing the second derivative,
///             r1 += istrxy * mucofu2 * u(1, i, j, q) + mucofuv * u(2, i, j, q) +
///                   istry * mucofuw * u(3, i, j, q);
///             r2 += mucofuv * u(1, i, j, q) + istrxy * mucofv2 * u(2, i, j, q) +
///                   istrx * mucofvw * u(3, i, j, q);
///             r3 += istry * mucofuw * u(1, i, j, q) +
///                   istrx * mucofvw * u(2, i, j, q) +
///                   istrxy * mucofw2 * u(3, i, j, q);
///           }
///
///           // Ghost point values, only nonzero for k=nk.
///           // 72 ops., tot=4011
///           mucofu2 = ghcof_no_gp(nk - k + 1) *
///                     ((2 * mu(i, j, nk) + la(i, j, nk)) * met(2, i, j, nk) *
///                          strx(i) * met(2, i, j, nk) * strx(i) +
///                      mu(i, j, nk) * (met(3, i, j, nk) * stry(j) *
///                                          met(3, i, j, nk) * stry(j) +
///                                      met(4, i, j, nk) * met(4, i, j, nk)));
///           mucofv2 = ghcof_no_gp(nk - k + 1) *
///                     ((2 * mu(i, j, nk) + la(i, j, nk)) * met(3, i, j, nk) *
///                          stry(j) * met(3, i, j, nk) * stry(j) +
///                      mu(i, j, nk) * (met(2, i, j, nk) * strx(i) *
///                                          met(2, i, j, nk) * strx(i) +
///                                      met(4, i, j, nk) * met(4, i, j, nk)));
///           mucofw2 =
///               ghcof_no_gp(nk - k + 1) *
///               ((2 * mu(i, j, nk) + la(i, j, nk)) * met(4, i, j, nk) *
///                    met(4, i, j, nk) +
///                mu(i, j, nk) *
///                    (met(2, i, j, nk) * strx(i) * met(2, i, j, nk) * strx(i) +
///                     met(3, i, j, nk) * stry(j) * met(3, i, j, nk) * stry(j)));
///           mucofuv = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
///                     met(2, i, j, nk) * met(3, i, j, nk);
///           mucofuw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
///                     met(2, i, j, nk) * met(4, i, j, nk);
///           mucofvw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *
///                     met(3, i, j, nk) * met(4, i, j, nk);
///           r1 += istrxy * mucofu2 * u(1, i, j, nk + 1) +
///                 mucofuv * u(2, i, j, nk + 1) +
///                 istry * mucofuw * u(3, i, j, nk + 1);
///           r2 += mucofuv * u(1, i, j, nk + 1) +
///                 istrxy * mucofv2 * u(2, i, j, nk + 1) +
///                 istrx * mucofvw * u(3, i, j, nk + 1);
///           r3 += istry * mucofuw * u(1, i, j, nk + 1) +
///                 istrx * mucofvw * u(2, i, j, nk + 1) +
///                 istrxy * mucofw2 * u(3, i, j, nk + 1);
///
///           // pq-derivatives (u-eq)
///           // 38 ops., tot=4049
///           r1 +=
///               c2 *
///                   (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
///                        (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
///                         c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
///                    mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
///                        (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
///                         c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +
///               c1 *
///                   (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
///                        (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
///                         c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
///                    mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
///                        (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
///                         c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));
///
///           // qp-derivatives (u-eq)
///           // 38 ops. tot=4087
///           r1 +=
///               c2 *
///                   (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                        (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
///                         c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
///                    la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                        (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
///                         c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +
///               c1 *
///                   (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                        (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
///                         c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
///                    la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                        (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
///                         c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));
///
///           // pq-derivatives (v-eq)
///           // 38 ops. , tot=4125
///           r2 +=
///               c2 *
///                   (la(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
///                        (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +
///                         c1 * (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -
///                    la(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
///                        (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +
///                         c1 * (u(1, i + 1, j - 2, k) - u(1, i - 1, j - 2, k)))) +
///               c1 *
///                   (la(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
///                        (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +
///                         c1 * (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -
///                    la(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
///                        (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +
///                         c1 * (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));
///
///           //* qp-derivatives (v-eq)
///           // 38 ops., tot=4163
///           r2 +=
///               c2 *
///                   (mu(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                        (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +
///                         c1 * (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -
///                    mu(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                        (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +
///                         c1 * (u(1, i - 2, j + 1, k) - u(1, i - 2, j - 1, k)))) +
///               c1 *
///                   (mu(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                        (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +
///                         c1 * (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -
///                    mu(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                        (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +
///                         c1 * (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));
///
///           // rp - derivatives
///           // 24*8 = 192 ops, tot=4355
///           float_sw4 dudrm2 = 0, dudrm1 = 0, dudrp1 = 0, dudrp2 = 0;
///           float_sw4 dvdrm2 = 0, dvdrm1 = 0, dvdrp1 = 0, dvdrp2 = 0;
///           float_sw4 dwdrm2 = 0, dwdrm1 = 0, dwdrp1 = 0, dwdrp2 = 0;
///           //#pragma unroll 8
///           for (int q = nk - 7; q <= nk; q++) {
///             dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 2, j, q);
///             dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 2, j, q);
///             dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 2, j, q);
///             dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 1, j, q);
///             dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 1, j, q);
///             dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 1, j, q);
///             dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 2, j, q);
///             dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 2, j, q);
///             dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 2, j, q);
///             dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 1, j, q);
///             dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 1, j, q);
///             dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 1, j, q);
///           }
///
///           // rp derivatives (u-eq)
///           // 67 ops, tot=4422
///           r1 += (c2 * ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
///                            met(2, i + 2, j, k) * met(1, i + 2, j, k) *
///                            strx(i + 2) * dudrp2 +
///                        la(i + 2, j, k) * met(3, i + 2, j, k) *
///                            met(1, i + 2, j, k) * dvdrp2 * stry(j) +
///                        la(i + 2, j, k) * met(4, i + 2, j, k) *
///                            met(1, i + 2, j, k) * dwdrp2 -
///                        ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
///                             met(2, i - 2, j, k) * met(1, i - 2, j, k) *
///                             strx(i - 2) * dudrm2 +
///                         la(i - 2, j, k) * met(3, i - 2, j, k) *
///                             met(1, i - 2, j, k) * dvdrm2 * stry(j) +
///                         la(i - 2, j, k) * met(4, i - 2, j, k) *
///                             met(1, i - 2, j, k) * dwdrm2)) +
///                  c1 * ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
///                            met(2, i + 1, j, k) * met(1, i + 1, j, k) *
///                            strx(i + 1) * dudrp1 +
///                        la(i + 1, j, k) * met(3, i + 1, j, k) *
///                            met(1, i + 1, j, k) * dvdrp1 * stry(j) +
///                        la(i + 1, j, k) * met(4, i + 1, j, k) *
///                            met(1, i + 1, j, k) * dwdrp1 -
///                        ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
///                             met(2, i - 1, j, k) * met(1, i - 1, j, k) *
///                             strx(i - 1) * dudrm1 +
///                         la(i - 1, j, k) * met(3, i - 1, j, k) *
///                             met(1, i - 1, j, k) * dvdrm1 * stry(j) +
///                         la(i - 1, j, k) * met(4, i - 1, j, k) *
///                             met(1, i - 1, j, k) * dwdrm1))) *
///                 istry;
///
///           // rp derivatives (v-eq)
///           // 42 ops, tot=4464
///           r2 +=
///               c2 * (mu(i + 2, j, k) * met(3, i + 2, j, k) *
///                         met(1, i + 2, j, k) * dudrp2 +
///                     mu(i + 2, j, k) * met(2, i + 2, j, k) *
///                         met(1, i + 2, j, k) * dvdrp2 * strx(i + 2) * istry -
///                     (mu(i - 2, j, k) * met(3, i - 2, j, k) *
///                          met(1, i - 2, j, k) * dudrm2 +
///                      mu(i - 2, j, k) * met(2, i - 2, j, k) *
///                          met(1, i - 2, j, k) * dvdrm2 * strx(i - 2) * istry)) +
///               c1 * (mu(i + 1, j, k) * met(3, i + 1, j, k) *
///                         met(1, i + 1, j, k) * dudrp1 +
///                     mu(i + 1, j, k) * met(2, i + 1, j, k) *
///                         met(1, i + 1, j, k) * dvdrp1 * strx(i + 1) * istry -
///                     (mu(i - 1, j, k) * met(3, i - 1, j, k) *
///                          met(1, i - 1, j, k) * dudrm1 +
///                      mu(i - 1, j, k) * met(2, i - 1, j, k) *
///                          met(1, i - 1, j, k) * dvdrm1 * strx(i - 1) * istry));
///
///           // rp derivatives (w-eq)
///           // 38 ops, tot=4502
///           r3 +=
///               istry * (c2 * (mu(i + 2, j, k) * met(4, i + 2, j, k) *
///                                  met(1, i + 2, j, k) * dudrp2 +
///                              mu(i + 2, j, k) * met(2, i + 2, j, k) *
///                                  met(1, i + 2, j, k) * dwdrp2 * strx(i + 2) -
///                              (mu(i - 2, j, k) * met(4, i - 2, j, k) *
///                                   met(1, i - 2, j, k) * dudrm2 +
///                               mu(i - 2, j, k) * met(2, i - 2, j, k) *
///                                   met(1, i - 2, j, k) * dwdrm2 * strx(i - 2))) +
///                        c1 * (mu(i + 1, j, k) * met(4, i + 1, j, k) *
///                                  met(1, i + 1, j, k) * dudrp1 +
///                              mu(i + 1, j, k) * met(2, i + 1, j, k) *
///                                  met(1, i + 1, j, k) * dwdrp1 * strx(i + 1) -
///                              (mu(i - 1, j, k) * met(4, i - 1, j, k) *
///                                   met(1, i - 1, j, k) * dudrm1 +
///                               mu(i - 1, j, k) * met(2, i - 1, j, k) *
///                                   met(1, i - 1, j, k) * dwdrm1 * strx(i - 1))));
///
///           // rq - derivatives
///           // 24*8 = 192 ops , tot=4694
///
///           dudrm2 = 0;
///           dudrm1 = 0;
///           dudrp1 = 0;
///           dudrp2 = 0;
///           dvdrm2 = 0;
///           dvdrm1 = 0;
///           dvdrp1 = 0;
///           dvdrp2 = 0;
///           dwdrm2 = 0;
///           dwdrm1 = 0;
///           dwdrp1 = 0;
///           dwdrp2 = 0;
///           //#pragma unroll 8
///           for (int q = nk - 7; q <= nk; q++) {
///             dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 2, q);
///             dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 2, q);
///             dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 2, q);
///             dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 1, q);
///             dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 1, q);
///             dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 1, q);
///             dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 2, q);
///             dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 2, q);
///             dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 2, q);
///             dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 1, q);
///             dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 1, q);
///             dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 1, q);
///           }
///
///           // rq derivatives (u-eq)
///           // 42 ops, tot=4736
///           r1 += c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
///                           met(1, i, j + 2, k) * dudrp2 * stry(j + 2) * istrx +
///                       mu(i, j + 2, k) * met(2, i, j + 2, k) *
///                           met(1, i, j + 2, k) * dvdrp2 -
///                       (mu(i, j - 2, k) * met(3, i, j - 2, k) *
///                            met(1, i, j - 2, k) * dudrm2 * stry(j - 2) * istrx +
///                        mu(i, j - 2, k) * met(2, i, j - 2, k) *
///                            met(1, i, j - 2, k) * dvdrm2)) +
///                 c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
///                           met(1, i, j + 1, k) * dudrp1 * stry(j + 1) * istrx +
///                       mu(i, j + 1, k) * met(2, i, j + 1, k) *
///                           met(1, i, j + 1, k) * dvdrp1 -
///                       (mu(i, j - 1, k) * met(3, i, j - 1, k) *
///                            met(1, i, j - 1, k) * dudrm1 * stry(j - 1) * istrx +
///                        mu(i, j - 1, k) * met(2, i, j - 1, k) *
///                            met(1, i, j - 1, k) * dvdrm1));
///
///           // rq derivatives (v-eq)
///           // 70 ops, tot=4806
///           r2 += c2 * (la(i, j + 2, k) * met(2, i, j + 2, k) *
///                           met(1, i, j + 2, k) * dudrp2 +
///                       (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *
///                           met(3, i, j + 2, k) * met(1, i, j + 2, k) * dvdrp2 *
///                           stry(j + 2) * istrx +
///                       la(i, j + 2, k) * met(4, i, j + 2, k) *
///                           met(1, i, j + 2, k) * dwdrp2 * istrx -
///                       (la(i, j - 2, k) * met(2, i, j - 2, k) *
///                            met(1, i, j - 2, k) * dudrm2 +
///                        (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *
///                            met(3, i, j - 2, k) * met(1, i, j - 2, k) * dvdrm2 *
///                            stry(j - 2) * istrx +
///                        la(i, j - 2, k) * met(4, i, j - 2, k) *
///                            met(1, i, j - 2, k) * dwdrm2 * istrx)) +
///                 c1 * (la(i, j + 1, k) * met(2, i, j + 1, k) *
///                           met(1, i, j + 1, k) * dudrp1 +
///                       (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *
///                           met(3, i, j + 1, k) * met(1, i, j + 1, k) * dvdrp1 *
///                           stry(j + 1) * istrx +
///                       la(i, j + 1, k) * met(4, i, j + 1, k) *
///                           met(1, i, j + 1, k) * dwdrp1 * istrx -
///                       (la(i, j - 1, k) * met(2, i, j - 1, k) *
///                            met(1, i, j - 1, k) * dudrm1 +
///                        (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *
///                            met(3, i, j - 1, k) * met(1, i, j - 1, k) * dvdrm1 *
///                            stry(j - 1) * istrx +
///                        la(i, j - 1, k) * met(4, i, j - 1, k) *
///                            met(1, i, j - 1, k) * dwdrm1 * istrx));
///
///           // rq derivatives (w-eq)
///           // 39 ops, tot=4845
///           r3 += (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *
///                            met(1, i, j + 2, k) * dwdrp2 * stry(j + 2) +
///                        mu(i, j + 2, k) * met(4, i, j + 2, k) *
///                            met(1, i, j + 2, k) * dvdrp2 -
///                        (mu(i, j - 2, k) * met(3, i, j - 2, k) *
///                             met(1, i, j - 2, k) * dwdrm2 * stry(j - 2) +
///                         mu(i, j - 2, k) * met(4, i, j - 2, k) *
///                             met(1, i, j - 2, k) * dvdrm2)) +
///                  c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *
///                            met(1, i, j + 1, k) * dwdrp1 * stry(j + 1) +
///                        mu(i, j + 1, k) * met(4, i, j + 1, k) *
///                            met(1, i, j + 1, k) * dvdrp1 -
///                        (mu(i, j - 1, k) * met(3, i, j - 1, k) *
///                             met(1, i, j - 1, k) * dwdrm1 * stry(j - 1) +
///                         mu(i, j - 1, k) * met(4, i, j - 1, k) *
///                             met(1, i, j - 1, k) * dvdrm1))) *
///                 istrx;
///
///           // pr and qr derivatives at once
///           // in loop: 8*(53+53+43) = 1192 ops, tot=6037
///           //#pragma unroll 8
///           for (int q = nk - 7; q <= nk; q++) {
///             // (u-eq)
///             // 53 ops
///             r1 -= bope(nk - k + 1, nk - q + 1) *
///                   (
///                       // pr
///                       (2 * mu(i, j, q) + la(i, j, q)) * met(2, i, j, q) *
///                           met(1, i, j, q) *
///                           (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
///                            c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
///                           strx(i) * istry +
///                       mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
///                            c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) +
///                       mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
///                            c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
///                           istry
///                       // qr
///                       + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
///                             (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
///                              c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) *
///                             stry(j) * istrx +
///                       la(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
///                            c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))));
///
///             // (v-eq)
///             // 53 ops
///             r2 -= bope(nk - k + 1, nk - q + 1) *
///                   (
///                       // pr
///                       la(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
///                            c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) +
///                       mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +
///                            c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) *
///                           strx(i) * istry
///                       // qr
///                       + mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
///                             (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +
///                              c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) +
///                       (2 * mu(i, j, q) + la(i, j, q)) * met(3, i, j, q) *
///                           met(1, i, j, q) *
///                           (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
///                            c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
///                           stry(j) * istrx +
///                       mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
///                            c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
///                           istrx);
///
///             // (w-eq)
///             // 43 ops
///             r3 -= bope(nk - k + 1, nk - q + 1) *
///                   (
///                       // pr
///                       la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +
///                            c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *
///                           istry +
///                       mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +
///                            c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *
///                           strx(i) * istry
///                       // qr
///                       + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *
///                             (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +
///                              c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *
///                             stry(j) * istrx +
///                       la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *
///                           (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +
///                            c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *
///                           istrx);
///           }
///
///           // 12 ops, tot=6049
///           lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
///           lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;
///           lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;
///
///       }

#ifndef RAJAPerf_Apps_SW4CK_KERNEL_5_HPP
#define RAJAPerf_Apps_SW4CK_KERNEL_5_HPP

using float_sw4 = double;

#define SW4CK_KERNEL_5_DATA_SETUP                           \
  float_sw4 a1 = 0;                                         \
  float_sw4 sgn = 1;                                        \
  if (op == '=') {                                          \
    a1 = 0;                                                 \
    sgn = 1;                                                \
  } else if (op == '+') {                                   \
    a1 = 1;                                                 \
    sgn = 1;                                                \
  } else if (op == '-') {                                   \
    a1 = 1;                                                 \
    sgn = -1;                                               \
  }                                                         \
                                                            \
  const float_sw4 i6 = 1.0 / 6;                             \
  const float_sw4 tf = 0.75;                                \
  const float_sw4 c1 = 2.0 / 3;                             \
  const float_sw4 c2 = -1.0 / 12;                           \
                                                            \
  const int ni = ilast - ifirst + 1;                        \
  const int nij = ni * (jlast - jfirst + 1);                \
  const int nijk = nij * (klast - kfirst + 1);              \
  const int base = -(ifirst + ni * jfirst + nij * kfirst);  \
  const int base3 = base - nijk;                            \
  const int base4 = base - nijk;                            \
  const int ifirst0 = ifirst;                               \
  const int jfirst0 = jfirst;                               \
                                                            \
  Real_ptr a_mu = m_a_mu;                                         \
  Real_ptr a_lambda = m_a_lambda;                                 \
  Real_ptr a_jac = m_a_jac;                                       \
  Real_ptr a_u = m_a_u;                                           \
  Real_ptr a_lu = m_a_lu;                                         \
  Real_ptr a_met = m_a_met;                                       \
  Real_ptr a_strx = m_a_strx;                                     \
  Real_ptr a_stry = m_a_stry;                                     \
  Real_ptr a_bope = m_a_bope;                                     \
  Real_ptr a_ghcof_no_gp = m_a_ghcof_no_gp;                       \
  Real_ptr a_acof_no_gp = m_a_acof_no_gp;

//not used
//Real_ptr a_acof = m_a_acof;
//Real_ptr a_ghcof = m_a_ghcof;

// Direct reuse of fortran code by these macro definitions:
#define mu(i, j, k) a_mu[base + (i) + ni * (j) + nij * (k)]
#define la(i, j, k) a_lambda[base + (i) + ni * (j) + nij * (k)]
#define jac(i, j, k) a_jac[base + (i) + ni * (j) + nij * (k)]
#define swck4_u(c, i, j, k) a_u[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define lu(c, i, j, k) a_lu[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define met(c, i, j, k) a_met[base4 + (i) + ni * (j) + nij * (k) + nijk * (c)]
#define strx(i) a_strx[i - ifirst0]
#define stry(j) a_stry[j - jfirst0]
#define acof(i, j, k) a_acof[(i - 1) + 6 * (j - 1) + 48 * (k - 1)]
#define bope(i, j) a_bope[i - 1 + 6 * (j - 1)]
#define ghcof(i) a_ghcof[i - 1]
#define acof_no_gp(i, j, k) a_acof_no_gp[(i - 1) + 6 * (j - 1) + 48 * (k - 1)]
#define ghcof_no_gp(i) a_ghcof_no_gp[i - 1]


// 5 ops
#define SW4CK_KERNEL_5_BODY_1                        \
  float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k); \
  float_sw4 istry = 1 / (stry(j));                   \
  float_sw4 istrx = 1 / (strx(i));                   \
  float_sw4 istrxy = istry * istrx;                  \
                                                     \
  float_sw4 r1 = 0, r2 = 0, r3 = 0;

// pp derivative (u) (u-eq)
// 53 ops, tot=58
#define SW4CK_KERNEL_5_BODY_2                                           \
  float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *            \
    met(1, i - 2, j, k) * met(1, i - 2, j, k) *                         \
    strx(i - 2);                                                        \
  float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *            \
    met(1, i - 1, j, k) * met(1, i - 1, j, k) *                         \
    strx(i - 1);                                                        \
  float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *  \
    met(1, i, j, k) * strx(i);                                          \
  float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *            \
    met(1, i + 1, j, k) * met(1, i + 1, j, k) *                         \
    strx(i + 1);                                                        \
  float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *            \
    met(1, i + 2, j, k) * met(1, i + 2, j, k) *                         \
    strx(i + 2);                                                        \
                                                                        \
  float_sw4 mux1 = cof2 - tf * (cof3 + cof1);                           \
  float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                     \
  float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                     \
  float_sw4 mux4 = cof4 - tf * (cof3 + cof5);                           \
                                                                        \
  r1 = r1 + i6 *                                                        \
    (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +                       \
     mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +                       \
     mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +                       \
     mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *                      \
    istry;

// qq derivative (u) (u-eq)
// 43 ops, tot=101
#define SW4CK_KERNEL_5_BODY_3                                           \
  cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) * \
    stry(j - 2);                                                        \
  cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) * \
    stry(j - 1);                                                        \
  cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);   \
  cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) * \
    stry(j + 1);                                                        \
  cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) * \
    stry(j + 2);                                                        \
                                                                        \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r1 = r1 + i6 *                                                        \
    (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +                       \
     mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +                       \
     mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +                       \
     mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *                      \
    istrx;


// pp derivative (v) (v-eq)
// 43 ops, tot=144
#define SW4CK_KERNEL_5_BODY_4                                           \
  cof1 = (mu(i - 2, j, k)) * met(1, i - 2, j, k) * met(1, i - 2, j, k) * \
    strx(i - 2);                                                        \
  cof2 = (mu(i - 1, j, k)) * met(1, i - 1, j, k) * met(1, i - 1, j, k) * \
    strx(i - 1);                                                        \
  cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * strx(i);   \
  cof4 = (mu(i + 1, j, k)) * met(1, i + 1, j, k) * met(1, i + 1, j, k) * \
    strx(i + 1);                                                        \
  cof5 = (mu(i + 2, j, k)) * met(1, i + 2, j, k) * met(1, i + 2, j, k) * \
    strx(i + 2);                                                        \
                                                                        \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r2 = r2 + i6 *                                                        \
    (mux1 * (u(2, i - 2, j, k) - u(2, i, j, k)) +                       \
     mux2 * (u(2, i - 1, j, k) - u(2, i, j, k)) +                       \
     mux3 * (u(2, i + 1, j, k) - u(2, i, j, k)) +                       \
     mux4 * (u(2, i + 2, j, k) - u(2, i, j, k))) *                      \
    istry;


// qq derivative (v) (v-eq)
// 53 ops, tot=197
#define SW4CK_KERNEL_5_BODY_5                                           \
  cof1 = (2 * mu(i, j - 2, k) + la(i, j - 2, k)) * met(1, i, j - 2, k) * \
    met(1, i, j - 2, k) * stry(j - 2);                                  \
  cof2 = (2 * mu(i, j - 1, k) + la(i, j - 1, k)) * met(1, i, j - 1, k) * \
    met(1, i, j - 1, k) * stry(j - 1);                                  \
  cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *            \
    met(1, i, j, k) * stry(j);                                          \
  cof4 = (2 * mu(i, j + 1, k) + la(i, j + 1, k)) * met(1, i, j + 1, k) * \
    met(1, i, j + 1, k) * stry(j + 1);                                  \
  cof5 = (2 * mu(i, j + 2, k) + la(i, j + 2, k)) * met(1, i, j + 2, k) * \
    met(1, i, j + 2, k) * stry(j + 2);                                  \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r2 = r2 + i6 *                                                        \
    (mux1 * (u(2, i, j - 2, k) - u(2, i, j, k)) +                       \
     mux2 * (u(2, i, j - 1, k) - u(2, i, j, k)) +                       \
     mux3 * (u(2, i, j + 1, k) - u(2, i, j, k)) +                       \
     mux4 * (u(2, i, j + 2, k) - u(2, i, j, k))) *                      \
    istrx;

// pp derivative (w) (w-eq)
// 43 ops, tot=240
#define SW4CK_KERNEL_5_BODY_6                                           \
  cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) * \
    stry(j - 2);                                                        \
  cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) * \
    stry(j - 1);                                                        \
  cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);   \
  cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) * \
    stry(j + 1);                                                        \
  cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) * \
    stry(j + 2);                                                        \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r3 = r3 + i6 *                                                        \
    (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +                       \
     mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +                       \
     mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +                       \
     mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *                      \
    istrx;

// qq derivative (w) (w-eq)
// 43 ops, tot=283
#define SW4CK_KERNEL_5_BODY_7                                           \
  cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) * met(1, i, j - 2, k) * \
    stry(j - 2);                                                        \
  cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) * met(1, i, j - 1, k) * \
    stry(j - 1);                                                        \
  cof3 = (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);   \
  cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) * met(1, i, j + 1, k) * \
    stry(j + 1);                                                        \
  cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) * met(1, i, j + 2, k) * \
    stry(j + 2);                                                        \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r3 = r3 + i6 *                                                        \
    (mux1 * (u(3, i, j - 2, k) - u(3, i, j, k)) +                       \
     mux2 * (u(3, i, j - 1, k) - u(3, i, j, k)) +                       \
     mux3 * (u(3, i, j + 1, k) - u(3, i, j, k)) +                       \
     mux4 * (u(3, i, j + 2, k) - u(3, i, j, k))) *                      \
    istrx;

// All rr-derivatives at once
// averaging the coefficient
// 54*8*8+25*8 = 3656 ops, tot=3939
#define SW4CK_KERNEL_5_BODY_8                                           \
  float_sw4 mucofu2, mucofuv, mucofuw, mucofvw, mucofv2, mucofw2;       \

// #if defined(MAGIC_SYNC) && defined(RAJA_GPU_DEVICE_COMPILE_PASS_ACTIVE)
//  __syncthreads();
// #endif

#define SW4CK_KERNEL_5_BODY_8_1                                         \
  /*#pragma unroll 8 */                                                 \
  for (int q = nk - 7; q <= nk; q++) {                                  \
    mucofu2 = 0;                                                        \
    mucofuv = 0;                                                        \
    mucofuw = 0;                                                        \
    mucofvw = 0;                                                        \
    mucofv2 = 0;                                                        \
    mucofw2 = 0;                                                        \

//  #ifdef AMD_UNROLL_FIX
//    #pragma unroll 8
//    #endif
#define SW4CK_KERNEL_5_BODY_8_2  \
for (int m = nk - 7; m <= nk; m++) {                                    \
    mucofu2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *         \
        ((2 * mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *            \
         strx(i) * met(2, i, j, m) * strx(i) +                          \
         mu(i, j, m) * (met(3, i, j, m) * stry(j) *                     \
                        met(3, i, j, m) * stry(j) +                     \
                        met(4, i, j, m) * met(4, i, j, m)));            \
      mucofv2 += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *       \
        ((2 * mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *            \
         stry(j) * met(3, i, j, m) * stry(j) +                          \
         mu(i, j, m) * (met(2, i, j, m) * strx(i) *                     \
                        met(2, i, j, m) * strx(i) +                     \
                        met(4, i, j, m) * met(4, i, j, m)));            \
      mucofw2 +=                                                        \
        acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *                \
        ((2 * mu(i, j, m) + la(i, j, m)) * met(4, i, j, m) *            \
         met(4, i, j, m) +                                              \
         mu(i, j, m) *                                                  \
         (met(2, i, j, m) * strx(i) * met(2, i, j, m) * strx(i) +       \
          met(3, i, j, m) * stry(j) * met(3, i, j, m) * stry(j)));      \
      mucofuv += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *       \
        (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *                 \
        met(3, i, j, m);                                                \
      mucofuw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *       \
        (mu(i, j, m) + la(i, j, m)) * met(2, i, j, m) *                 \
        met(4, i, j, m);                                                \
      mucofvw += acof_no_gp(nk - k + 1, nk - q + 1, nk - m + 1) *       \
        (mu(i, j, m) + la(i, j, m)) * met(3, i, j, m) *                 \
        met(4, i, j, m);                                                \
    }                                                                   \
                                                                        \
  /* Computing the second derivative, */                                \
  r1 += istrxy * mucofu2 * u(1, i, j, q) + mucofuv * u(2, i, j, q) +    \
  istry * mucofuw * u(3, i, j, q);                                      \
  r2 += mucofuv * u(1, i, j, q) + istrxy * mucofv2 * u(2, i, j, q) +    \
  istrx * mucofvw * u(3, i, j, q);                                      \
  r3 += istry * mucofuw * u(1, i, j, q) +                               \
  istrx * mucofvw * u(2, i, j, q) +                                     \
  istrxy * mucofw2 * u(3, i, j, q);                                     \
  }

// Ghost point values, only nonzero for k=nk.
// 72 ops., tot=4011
#define SW4CK_KERNEL_5_BODY_9                                           \
  mucofu2 = ghcof_no_gp(nk - k + 1) *                                   \
    ((2 * mu(i, j, nk) + la(i, j, nk)) * met(2, i, j, nk) *             \
    strx(i) * met(2, i, j, nk) * strx(i) +                              \
    mu(i, j, nk) * (met(3, i, j, nk) * stry(j) *                        \
                    met(3, i, j, nk) * stry(j) +                        \
                    met(4, i, j, nk) * met(4, i, j, nk)));              \
  mucofv2 = ghcof_no_gp(nk - k + 1) *                                   \
    ((2 * mu(i, j, nk) + la(i, j, nk)) * met(3, i, j, nk) *             \
     stry(j) * met(3, i, j, nk) * stry(j) +                             \
     mu(i, j, nk) * (met(2, i, j, nk) * strx(i) *                       \
                     met(2, i, j, nk) * strx(i) +                       \
                     met(4, i, j, nk) * met(4, i, j, nk)));             \
  mucofw2 =                                                             \
    ghcof_no_gp(nk - k + 1) *                                           \
    ((2 * mu(i, j, nk) + la(i, j, nk)) * met(4, i, j, nk) *             \
     met(4, i, j, nk) +                                                 \
     mu(i, j, nk) *                                                     \
     (met(2, i, j, nk) * strx(i) * met(2, i, j, nk) * strx(i) +         \
      met(3, i, j, nk) * stry(j) * met(3, i, j, nk) * stry(j)));        \
  mucofuv = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *   \
    met(2, i, j, nk) * met(3, i, j, nk);                                \
  mucofuw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *   \
    met(2, i, j, nk) * met(4, i, j, nk);                                \
  mucofvw = ghcof_no_gp(nk - k + 1) * (mu(i, j, nk) + la(i, j, nk)) *   \
    met(3, i, j, nk) * met(4, i, j, nk);                                \
  r1 += istrxy * mucofu2 * u(1, i, j, nk + 1) +                         \
    mucofuv * u(2, i, j, nk + 1) +                                      \
    istry * mucofuw * u(3, i, j, nk + 1);                               \
  r2 += mucofuv * u(1, i, j, nk + 1) +                                  \
    istrxy * mucofv2 * u(2, i, j, nk + 1) +                             \
    istrx * mucofvw * u(3, i, j, nk + 1);                               \
  r3 += istry * mucofuw * u(1, i, j, nk + 1) +                          \
    istrx * mucofvw * u(2, i, j, nk + 1) +                              \
    istrxy * mucofw2 * u(3, i, j, nk + 1);

// pq-derivatives (u-eq)
// 38 ops., tot=4049
#define SW4CK_KERNEL_5_BODY_10                                           \
  r1 +=                                                                 \
    c2 *                                                                \
    (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *      \
     (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +            \
      c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -           \
     mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *      \
     (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +            \
      c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +          \
    c1 *                                                                \
    (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *      \
     (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +            \
      c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -           \
     mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *      \
     (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +            \
      c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

// qp-derivatives (u-eq)
// 38 ops. tot=4087
#define SW4CK_KERNEL_5_BODY_11                  \
  r1 +=                                         \
    c2 *                                                                \
    (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *      \
     (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +            \
      c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -           \
     mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *      \
     (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +            \
      c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +          \
    c1 *                                                                \
    (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *      \
     (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +            \
      c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -           \
     mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *      \
     (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +            \
      c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

// pq-derivatives (v-eq)
// 38 ops. , tot=4125
#define SW4CK_KERNEL_5_BODY_12                                          \
  r2 +=                                                                 \
    c2 *                                                                \
    (la(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *      \
     (c2 * (u(1, i + 2, j + 2, k) - u(1, i - 2, j + 2, k)) +            \
      c1 * (u(1, i + 1, j + 2, k) - u(1, i - 1, j + 2, k))) -           \
     la(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *      \
     (c2 * (u(1, i + 2, j - 2, k) - u(1, i - 2, j - 2, k)) +            \
      c1 * (u(1, i + 1, j - 2, k) - u(1, i - 1, j - 2, k)))) +          \
    c1 *                                                                \
    (la(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *      \
     (c2 * (u(1, i + 2, j + 1, k) - u(1, i - 2, j + 1, k)) +            \
      c1 * (u(1, i + 1, j + 1, k) - u(1, i - 1, j + 1, k))) -           \
     la(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *      \
     (c2 * (u(1, i + 2, j - 1, k) - u(1, i - 2, j - 1, k)) +            \
      c1 * (u(1, i + 1, j - 1, k) - u(1, i - 1, j - 1, k))));


// qp-derivatives (v-eq)
// 38 ops., tot=4163
#define SW4CK_KERNEL_5_BODY_13                                          \
  r2 +=                                                                 \
    c2 *                                                                \
    (mu(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *      \
     (c2 * (u(1, i + 2, j + 2, k) - u(1, i + 2, j - 2, k)) +            \
      c1 * (u(1, i + 2, j + 1, k) - u(1, i + 2, j - 1, k))) -           \
     mu(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *      \
     (c2 * (u(1, i - 2, j + 2, k) - u(1, i - 2, j - 2, k)) +            \
      c1 * (u(1, i - 2, j + 1, k) - u(1, i - 2, j - 1, k)))) +          \
    c1 *                                                                \
    (mu(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *      \
     (c2 * (u(1, i + 1, j + 2, k) - u(1, i + 1, j - 2, k)) +            \
      c1 * (u(1, i + 1, j + 1, k) - u(1, i + 1, j - 1, k))) -           \
     mu(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *      \
     (c2 * (u(1, i - 1, j + 2, k) - u(1, i - 1, j - 2, k)) +            \
      c1 * (u(1, i - 1, j + 1, k) - u(1, i - 1, j - 1, k))));

// rp - derivatives
// 24*8 = 192 ops, tot=4355
#define SW4CK_KERNEL_5_BODY_14                                      \
  float_sw4 dudrm2 = 0, dudrm1 = 0, dudrp1 = 0, dudrp2 = 0;         \
  float_sw4 dvdrm2 = 0, dvdrm1 = 0, dvdrp1 = 0, dvdrp2 = 0;         \
  float_sw4 dwdrm2 = 0, dwdrm1 = 0, dwdrp1 = 0, dwdrp2 = 0;         \
  /*#pragma unroll 8 */                                                 \
  for (int q = nk - 7; q <= nk; q++) {                                  \
    dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 2, j, q);         \
    dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 2, j, q);         \
    dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 2, j, q);         \
    dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i - 1, j, q);         \
    dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i - 1, j, q);         \
    dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i - 1, j, q);         \
    dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 2, j, q);         \
    dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 2, j, q);         \
    dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 2, j, q);         \
    dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i + 1, j, q);         \
    dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i + 1, j, q);         \
    dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i + 1, j, q);         \
  }

// rp derivatives (u-eq)
// 67 ops, tot=4422
#define SW4CK_KERNEL_5_BODY_15 \
  r1 += (c2 * ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *               \
               met(2, i + 2, j, k) * met(1, i + 2, j, k) *             \
               strx(i + 2) * dudrp2 +                                  \
               la(i + 2, j, k) * met(3, i + 2, j, k) *                 \
               met(1, i + 2, j, k) * dvdrp2 * stry(j) +                \
               la(i + 2, j, k) * met(4, i + 2, j, k) *                 \
               met(1, i + 2, j, k) * dwdrp2 -                          \
               ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *               \
                met(2, i - 2, j, k) * met(1, i - 2, j, k) *             \
                strx(i - 2) * dudrm2 +                                  \
                la(i - 2, j, k) * met(3, i - 2, j, k) *                 \
                met(1, i - 2, j, k) * dvdrm2 * stry(j) +                \
                la(i - 2, j, k) * met(4, i - 2, j, k) *                 \
                met(1, i - 2, j, k) * dwdrm2)) +                        \
         c1 * ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *                \
               met(2, i + 1, j, k) * met(1, i + 1, j, k) *              \
               strx(i + 1) * dudrp1 +                                   \
               la(i + 1, j, k) * met(3, i + 1, j, k) *                  \
               met(1, i + 1, j, k) * dvdrp1 * stry(j) +                 \
               la(i + 1, j, k) * met(4, i + 1, j, k) *                  \
               met(1, i + 1, j, k) * dwdrp1 -                           \
               ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *               \
                met(2, i - 1, j, k) * met(1, i - 1, j, k) *             \
                strx(i - 1) * dudrm1 +                                  \
                la(i - 1, j, k) * met(3, i - 1, j, k) *                 \
                met(1, i - 1, j, k) * dvdrm1 * stry(j) +                \
                la(i - 1, j, k) * met(4, i - 1, j, k) *                 \
                met(1, i - 1, j, k) * dwdrm1))) *                       \
    istry;

// rp derivatives (v-eq)
// 42 ops, tot=4464
#define SW4CK_KERNEL_5_BODY_16                                          \
  r2 +=                                                                 \
    c2 * (mu(i + 2, j, k) * met(3, i + 2, j, k) *                       \
          met(1, i + 2, j, k) * dudrp2 +                                \
          mu(i + 2, j, k) * met(2, i + 2, j, k) *                       \
          met(1, i + 2, j, k) * dvdrp2 * strx(i + 2) * istry -          \
          (mu(i - 2, j, k) * met(3, i - 2, j, k) *                      \
           met(1, i - 2, j, k) * dudrm2 +                               \
           mu(i - 2, j, k) * met(2, i - 2, j, k) *                      \
           met(1, i - 2, j, k) * dvdrm2 * strx(i - 2) * istry)) +       \
    c1 * (mu(i + 1, j, k) * met(3, i + 1, j, k) *                       \
          met(1, i + 1, j, k) * dudrp1 +                                \
          mu(i + 1, j, k) * met(2, i + 1, j, k) *                       \
          met(1, i + 1, j, k) * dvdrp1 * strx(i + 1) * istry -          \
          (mu(i - 1, j, k) * met(3, i - 1, j, k) *                      \
           met(1, i - 1, j, k) * dudrm1 +                               \
           mu(i - 1, j, k) * met(2, i - 1, j, k) *                      \
           met(1, i - 1, j, k) * dvdrm1 * strx(i - 1) * istry));

// rp derivatives (w-eq)
// 38 ops, tot=4502
#define SW4CK_KERNEL_5_BODY_17                                          \
  r3 +=                                                                 \
    istry * (c2 * (mu(i + 2, j, k) * met(4, i + 2, j, k) *              \
                   met(1, i + 2, j, k) * dudrp2 +                       \
                   mu(i + 2, j, k) * met(2, i + 2, j, k) *              \
                   met(1, i + 2, j, k) * dwdrp2 * strx(i + 2) -         \
                   (mu(i - 2, j, k) * met(4, i - 2, j, k) *             \
                    met(1, i - 2, j, k) * dudrm2 +                      \
                    mu(i - 2, j, k) * met(2, i - 2, j, k) *             \
                    met(1, i - 2, j, k) * dwdrm2 * strx(i - 2))) +      \
             c1 * (mu(i + 1, j, k) * met(4, i + 1, j, k) *              \
                   met(1, i + 1, j, k) * dudrp1 +                       \
                   mu(i + 1, j, k) * met(2, i + 1, j, k) *              \
                   met(1, i + 1, j, k) * dwdrp1 * strx(i + 1) -         \
                   (mu(i - 1, j, k) * met(4, i - 1, j, k) *             \
                    met(1, i - 1, j, k) * dudrm1 +                      \
                    mu(i - 1, j, k) * met(2, i - 1, j, k) *             \
                    met(1, i - 1, j, k) * dwdrm1 * strx(i - 1))));


// rq - derivatives
// 24*8 = 192 ops , tot=4694
#define SW4CK_KERNEL_5_BODY_18                                          \
  dudrm2 = 0;                                                           \
  dudrm1 = 0;                                                           \
  dudrp1 = 0;                                                           \
  dudrp2 = 0;                                                           \
  dvdrm2 = 0;                                                           \
  dvdrm1 = 0;                                                           \
  dvdrp1 = 0;                                                           \
  dvdrp2 = 0;                                                           \
  dwdrm2 = 0;                                                           \
  dwdrm1 = 0;                                                           \
  dwdrp1 = 0;                                                           \
  dwdrp2 = 0;                                                           \
  /* #pragma unroll 8 */                                                \
  for (int q = nk - 7; q <= nk; q++) {                                  \
    dudrm2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 2, q);         \
    dvdrm2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 2, q);         \
    dwdrm2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 2, q);         \
    dudrm1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j - 1, q);         \
    dvdrm1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j - 1, q);         \
    dwdrm1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j - 1, q);         \
    dudrp2 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 2, q);         \
    dvdrp2 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 2, q);         \
    dwdrp2 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 2, q);         \
    dudrp1 -= bope(nk - k + 1, nk - q + 1) * u(1, i, j + 1, q);         \
    dvdrp1 -= bope(nk - k + 1, nk - q + 1) * u(2, i, j + 1, q);         \
    dwdrp1 -= bope(nk - k + 1, nk - q + 1) * u(3, i, j + 1, q);         \
  }

// rq derivatives (u-eq)
// 42 ops, tot=4736
#define SW4CK_KERNEL_5_BODY_19                                          \
  r1 += c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *                   \
              met(1, i, j + 2, k) * dudrp2 * stry(j + 2) * istrx +      \
              mu(i, j + 2, k) * met(2, i, j + 2, k) *                   \
              met(1, i, j + 2, k) * dvdrp2 -                            \
              (mu(i, j - 2, k) * met(3, i, j - 2, k) *                  \
               met(1, i, j - 2, k) * dudrm2 * stry(j - 2) * istrx +     \
               mu(i, j - 2, k) * met(2, i, j - 2, k) *                  \
               met(1, i, j - 2, k) * dvdrm2)) +                         \
    c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *                       \
          met(1, i, j + 1, k) * dudrp1 * stry(j + 1) * istrx +          \
          mu(i, j + 1, k) * met(2, i, j + 1, k) *                       \
          met(1, i, j + 1, k) * dvdrp1 -                                \
          (mu(i, j - 1, k) * met(3, i, j - 1, k) *                      \
           met(1, i, j - 1, k) * dudrm1 * stry(j - 1) * istrx +         \
           mu(i, j - 1, k) * met(2, i, j - 1, k) *                      \
           met(1, i, j - 1, k) * dvdrm1));

// rq derivatives (v-eq)
// 70 ops, tot=4806
#define SW4CK_KERNEL_5_BODY_20                                          \
  r2 += c2 * (la(i, j + 2, k) * met(2, i, j + 2, k) *                   \
              met(1, i, j + 2, k) * dudrp2 +                            \
              (2 * mu(i, j + 2, k) + la(i, j + 2, k)) *                 \
              met(3, i, j + 2, k) * met(1, i, j + 2, k) * dvdrp2 *      \
              stry(j + 2) * istrx +                                     \
              la(i, j + 2, k) * met(4, i, j + 2, k) *                   \
              met(1, i, j + 2, k) * dwdrp2 * istrx -                    \
              (la(i, j - 2, k) * met(2, i, j - 2, k) *                  \
                       met(1, i, j - 2, k) * dudrm2 +                   \
               (2 * mu(i, j - 2, k) + la(i, j - 2, k)) *                \
               met(3, i, j - 2, k) * met(1, i, j - 2, k) * dvdrm2 *     \
               stry(j - 2) * istrx +                                    \
               la(i, j - 2, k) * met(4, i, j - 2, k) *                  \
               met(1, i, j - 2, k) * dwdrm2 * istrx)) +                 \
    c1 * (la(i, j + 1, k) * met(2, i, j + 1, k) *                       \
          met(1, i, j + 1, k) * dudrp1 +                                \
          (2 * mu(i, j + 1, k) + la(i, j + 1, k)) *                     \
          met(3, i, j + 1, k) * met(1, i, j + 1, k) * dvdrp1 *          \
          stry(j + 1) * istrx +                                         \
          la(i, j + 1, k) * met(4, i, j + 1, k) *                       \
          met(1, i, j + 1, k) * dwdrp1 * istrx -                        \
          (la(i, j - 1, k) * met(2, i, j - 1, k) *                      \
           met(1, i, j - 1, k) * dudrm1 +                               \
           (2 * mu(i, j - 1, k) + la(i, j - 1, k)) *                    \
           met(3, i, j - 1, k) * met(1, i, j - 1, k) * dvdrm1 *         \
           stry(j - 1) * istrx +                                        \
           la(i, j - 1, k) * met(4, i, j - 1, k) *                      \
           met(1, i, j - 1, k) * dwdrm1 * istrx));

// rq derivatives (w-eq)
// 39 ops, tot=4845
#define SW4CK_KERNEL_5_BODY_21                                          \
  r3 += (c2 * (mu(i, j + 2, k) * met(3, i, j + 2, k) *                  \
               met(1, i, j + 2, k) * dwdrp2 * stry(j + 2) +             \
               mu(i, j + 2, k) * met(4, i, j + 2, k) *                  \
               met(1, i, j + 2, k) * dvdrp2 -                           \
               (mu(i, j - 2, k) * met(3, i, j - 2, k) *                 \
                met(1, i, j - 2, k) * dwdrm2 * stry(j - 2) +            \
                mu(i, j - 2, k) * met(4, i, j - 2, k) *                 \
                met(1, i, j - 2, k) * dvdrm2)) +                        \
         c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) *                  \
               met(1, i, j + 1, k) * dwdrp1 * stry(j + 1) +             \
               mu(i, j + 1, k) * met(4, i, j + 1, k) *                  \
               met(1, i, j + 1, k) * dvdrp1 -                           \
               (mu(i, j - 1, k) * met(3, i, j - 1, k) *                 \
                met(1, i, j - 1, k) * dwdrm1 * stry(j - 1) +            \
                mu(i, j - 1, k) * met(4, i, j - 1, k) *                 \
                met(1, i, j - 1, k) * dvdrm1))) *                       \
    istrx;

// pr and qr derivatives at once
// in loop: 8*(53+53+43) = 1192 ops, tot=6037
#define SW4CK_KERNEL_5_BODY_22                                          \
  /* #pragma unroll 8 */                                                \
  for (int q = nk - 7; q <= nk; q++) {                                  \
  /* (u-eq) */                                                          \
  /* 53 ops */                                                          \
  r1 -= bope(nk - k + 1, nk - q + 1) *                                  \
    (                                                                   \
    /* pr */                                                            \
    (2 * mu(i, j, q) + la(i, j, q)) * met(2, i, j, q) *                 \
    met(1, i, j, q) *                                                   \
(c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +                         \
 c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *                        \
    strx(i) * istry +                                                   \
    mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *                   \
    (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +                     \
     c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) +                    \
    mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *                   \
    (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +                     \
     c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *                    \
    istry                                                               \
    /* qr */                                                            \
    + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *                 \
    (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +                     \
     c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) *                    \
    stry(j) * istrx +                                                   \
    la(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *                   \
    (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +                     \
     c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))));                    \
                                                                        \
  /* (v-eq) */                                                          \
  /* 53 ops */                                                          \
  r2 -= bope(nk - k + 1, nk - q + 1) *                                  \
    (                                                                   \
     /* pr */                                                           \
     la(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +                    \
      c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) +                   \
     mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(2, i + 2, j, q) - u(2, i - 2, j, q)) +                    \
      c1 * (u(2, i + 1, j, q) - u(2, i - 1, j, q))) *                   \
     strx(i) * istry                                                    \
     /* qr */                                                           \
     + mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *                \
     (c2 * (u(1, i, j + 2, q) - u(1, i, j - 2, q)) +                    \
      c1 * (u(1, i, j + 1, q) - u(1, i, j - 1, q))) +                   \
     (2 * mu(i, j, q) + la(i, j, q)) * met(3, i, j, q) *                \
     met(1, i, j, q) *                                                  \
     (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +                    \
      c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *                   \
     stry(j) * istrx +                                                  \
     mu(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +                    \
      c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *                   \
     istrx);                                                            \
                                                                        \
  /* (w-eq) */                                                          \
  /* 43 ops */                                                          \
  r3 -= bope(nk - k + 1, nk - q + 1) *                                  \
    (                                                                   \
     /* pr */                                                           \
     la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(1, i + 2, j, q) - u(1, i - 2, j, q)) +                    \
      c1 * (u(1, i + 1, j, q) - u(1, i - 1, j, q))) *                   \
     istry +                                                            \
     mu(i, j, q) * met(2, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(3, i + 2, j, q) - u(3, i - 2, j, q)) +                    \
      c1 * (u(3, i + 1, j, q) - u(3, i - 1, j, q))) *                   \
     strx(i) * istry                                                    \
     /* qr */                                                           \
     + mu(i, j, q) * met(3, i, j, q) * met(1, i, j, q) *                \
     (c2 * (u(3, i, j + 2, q) - u(3, i, j - 2, q)) +                    \
      c1 * (u(3, i, j + 1, q) - u(3, i, j - 1, q))) *                   \
     stry(j) * istrx +                                                  \
     la(i, j, q) * met(4, i, j, q) * met(1, i, j, q) *                  \
     (c2 * (u(2, i, j + 2, q) - u(2, i, j - 2, q)) +                    \
      c1 * (u(2, i, j + 1, q) - u(2, i, j - 1, q))) *                   \
     istrx);                                                            \
          }

// 12 ops, tot=6049
#define SW4CK_KERNEL_5_BODY_23                                          \
  lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;               \
  lu(2, i, j, k) = a1 * lu(2, i, j, k) + sgn * r2 * ijac;               \
  lu(3, i, j, k) = a1 * lu(3, i, j, k) + sgn * r3 * ijac;

#include "common/KernelBase.hpp"

  namespace rajaperf {
  class RunParams;

  namespace apps {

  class SW4CK_KERNEL_5 : public KernelBase {
  public:
    SW4CK_KERNEL_5(const RunParams &params);

    ~SW4CK_KERNEL_5();

    void setUp(VariantID vid, size_t tune_idx);
    void updateChecksum(VariantID vid, size_t tune_idx);
    void tearDown(VariantID vid, size_t tune_idx);

    void runSeqVariant(VariantID vid, size_t tune_idx);
    void runOpenMPVariant(VariantID vid, size_t tune_idx);
    void runCudaVariant(VariantID vid, size_t tune_idx);
    void runHipVariant(VariantID vid, size_t tune_idx);
    void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

    void setCudaTuningDefinitions(VariantID vid);
    void setHipTuningDefinitions(VariantID vid);
    template <size_t block_size> void runCudaVariantImpl(VariantID vid);
    template <size_t block_size> void runHipVariantImpl(VariantID vid);

  private:
    static const size_t default_gpu_block_size = 256;
    using gpu_block_sizes_type = integer::list_type<default_gpu_block_size>;

    Real_ptr m_a_mu;
    Real_ptr m_a_lambda;
    Real_ptr m_a_jac;
    Real_ptr m_a_u;
    Real_ptr m_a_lu;
    Real_ptr m_a_met;
    Real_ptr m_a_strx;
    Real_ptr m_a_stry;
    Real_ptr m_a_acof;
    Real_ptr m_a_bope;
    Real_ptr m_a_ghcof;
    Real_ptr m_a_acof_no_gp;
    Real_ptr m_a_ghcof_no_gp;

  };

  } // end namespace apps
  } // end namespace rajaperf

#endif // closing endif for header file include guard
