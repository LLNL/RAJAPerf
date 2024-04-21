//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SW4CK_KERNEL_2 kernel reference implementation:
/// https://github.com/LLNL/SW4CK
///
///   for (int k = kstart; k <= klast - 2; k++)
///     for (int j = jfirst + 2; j <= jlast - 2; j++)
///       for (int i = ifirst + 2; i <= ilast - 2; i++) {
///
///         // 5 ops
///         float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
///         float_sw4 istry = 1 / (stry(j));
///         float_sw4 istrx = 1 / (strx(i));
///         float_sw4 istrxy = istry * istrx;
///
///         float_sw4 r1 = 0;
///
///         // pp derivative (u)
///         // 53 ops, tot=58
///         float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
///                          met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                          strx(i - 2);
///         float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
///                          met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                          strx(i - 1);
///         float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
///                          met(1, i, j, k) * strx(i);
///         float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
///                          met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                          strx(i + 1);
///         float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
///                          met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                          strx(i + 2);
///         float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
///         float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///         float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///         float_sw4 mux4 = cof4 - tf * (cof3 + cof5);
///
///         r1 += i6 *
///               (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
///                mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
///                mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
///                mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
///               istry;
///         // qq derivative (u)
///         // 43 ops, tot=101
///         {
///           float_sw4 cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
///                            met(1, i, j - 2, k) * stry(j - 2);
///           float_sw4 cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
///                            met(1, i, j - 1, k) * stry(j - 1);
///           float_sw4 cof3 =
///               (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
///           float_sw4 cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
///                            met(1, i, j + 1, k) * stry(j + 1);
///           float_sw4 cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
///                            met(1, i, j + 2, k) * stry(j + 2);
///           float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
///           float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           float_sw4 mux4 = cof4 - tf * (cof3 + cof5);
///
///           r1 += i6 *
///                 (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
///                  mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
///                  mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
///                  mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
///                 istrx;
///         }
/// #ifdef MAGIC_SYNC
///         __syncthreads();
/// #endif
///         // rr derivative (u)
///         // 5*11+14+14=83 ops, tot=184
///         {
///           float_sw4 cof1 =
///               (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
///                   strx(i) * met(2, i, j, k - 2) * strx(i) +
///               mu(i, j, k - 2) * (met(3, i, j, k - 2) * stry(j) *
///                                      met(3, i, j, k - 2) * stry(j) +
///                                  met(4, i, j, k - 2) * met(4, i, j, k - 2));
///           float_sw4 cof2 =
///               (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
///                   strx(i) * met(2, i, j, k - 1) * strx(i) +
///               mu(i, j, k - 1) * (met(3, i, j, k - 1) * stry(j) *
///                                      met(3, i, j, k - 1) * stry(j) +
///                                  met(4, i, j, k - 1) * met(4, i, j, k - 1));
///           float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) *
///                                strx(i) * met(2, i, j, k) * strx(i) +
///                            mu(i, j, k) * (met(3, i, j, k) * stry(j) *
///                                               met(3, i, j, k) * stry(j) +
///                                           met(4, i, j, k) * met(4, i, j, k));
///           float_sw4 cof4 =
///               (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
///                   strx(i) * met(2, i, j, k + 1) * strx(i) +
///               mu(i, j, k + 1) * (met(3, i, j, k + 1) * stry(j) *
///                                      met(3, i, j, k + 1) * stry(j) +
///                                  met(4, i, j, k + 1) * met(4, i, j, k + 1));
///           float_sw4 cof5 =
///               (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
///                   strx(i) * met(2, i, j, k + 2) * strx(i) +
///               mu(i, j, k + 2) * (met(3, i, j, k + 2) * stry(j) *
///                                      met(3, i, j, k + 2) * stry(j) +
///                                  met(4, i, j, k + 2) * met(4, i, j, k + 2));
///
///           float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
///           float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///           float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///           float_sw4 mux4 = cof4 - tf * (cof3 + cof5);
///
///           r1 += i6 *
///                 (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
///                  mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
///                  mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
///                  mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *
///                 istrxy;
///         }
///         // rr derivative (v)
///         // 42 ops, tot=226
///         cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
///                met(3, i, j, k - 2);
///         cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
///                met(3, i, j, k - 1);
///         cof3 = (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
///         cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
///                met(3, i, j, k + 1);
///         cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
///                met(3, i, j, k + 2);
///         mux1 = cof2 - tf * (cof3 + cof1);
///         mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///         mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///         mux4 = cof4 - tf * (cof3 + cof5);
///
///         r1 += i6 * (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
///                     mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
///                     mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
///                     mux4 * (u(2, i, j, k + 2) - u(2, i, j, k)));
///
///         // rr derivative (w)
///         // 43 ops, tot=269
///         cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
///                met(4, i, j, k - 2);
///         cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
///                met(4, i, j, k - 1);
///         cof3 = (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);
///         cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
///                met(4, i, j, k + 1);
///         cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
///                met(4, i, j, k + 2);
///         mux1 = cof2 - tf * (cof3 + cof1);
///         mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
///         mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
///         mux4 = cof4 - tf * (cof3 + cof5);
///
///         r1 += i6 *
///               (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
///                mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
///                mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
///                mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
///               istry;
///
///         // pq-derivatives
///         // 38 ops, tot=307
///         r1 +=
///             c2 * (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
///                       (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
///                        c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
///                   mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
///                       (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
///                        c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +
///             c1 * (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
///                       (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
///                        c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
///                   mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
///                       (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
///                        c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));
///
///         // qp-derivatives
///         // 38 ops, tot=345
///         r1 +=
///             c2 * (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
///                       (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
///                        c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
///                   la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
///                       (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
///                        c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +
///             c1 * (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
///                       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
///                        c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
///                   la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
///                       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
///                        c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));
///
///         // pr-derivatives
///         // 130 ops., tot=475
///         r1 +=
///             c2 * ((2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
///                       met(2, i, j, k + 2) * met(1, i, j, k + 2) *
///                       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
///                        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *
///                       strx(i) * istry +
///                   mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
///                       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
///                        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) +
///                   mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
///                       (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +
///                        c1 * (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *
///                       istry -
///                   ((2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
///                        met(2, i, j, k - 2) * met(1, i, j, k - 2) *
///                        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
///                         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) *
///                        strx(i) * istry +
///                    mu(i, j, k - 2) * met(3, i, j, k - 2) * met(1, i, j, k - 2) *
///                        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
///                         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) +
///                    mu(i, j, k - 2) * met(4, i, j, k - 2) * met(1, i, j, k - 2) *
///                        (c2 * (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +
///                         c1 * (u(3, i + 1, j, k - 2) - u(3, i - 1, j, k - 2))) *
///                        istry)) +
///             c1 * ((2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
///                       met(2, i, j, k + 1) * met(1, i, j, k + 1) *
///                       (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
///                        c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *
///                       strx(i) * istry +
///                   mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
///                       (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
///                        c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) +
///                   mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
///                       (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +
///                        c1 * (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *
///                       istry -
///                   ((2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
///                        met(2, i, j, k - 1) * met(1, i, j, k - 1) *
///                        (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
///                         c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) *
///                        strx(i) * istry +
///                    mu(i, j, k - 1) * met(3, i, j, k - 1) * met(1, i, j, k - 1) *
///                        (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
///                         c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) +
///                    mu(i, j, k - 1) * met(4, i, j, k - 1) * met(1, i, j, k - 1) *
///                        (c2 * (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +
///                         c1 * (u(3, i + 1, j, k - 1) - u(3, i - 1, j, k - 1))) *
///                        istry));
///
///         // rp derivatives
///         // 130 ops, tot=605
///         r1 +=
///             (c2 *
///                  ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
///                       met(2, i + 2, j, k) * met(1, i + 2, j, k) *
///                       (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
///                        c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) *
///                       strx(i + 2) +
///                   la(i + 2, j, k) * met(3, i + 2, j, k) * met(1, i + 2, j, k) *
///                       (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
///                        c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
///                       stry(j) +
///                   la(i + 2, j, k) * met(4, i + 2, j, k) * met(1, i + 2, j, k) *
///                       (c2 * (u(3, i + 2, j, k + 2) - u(3, i + 2, j, k - 2)) +
///                        c1 * (u(3, i + 2, j, k + 1) - u(3, i + 2, j, k - 1))) -
///                   ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
///                        met(2, i - 2, j, k) * met(1, i - 2, j, k) *
///                        (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
///                         c1 * (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) *
///                        strx(i - 2) +
///                    la(i - 2, j, k) * met(3, i - 2, j, k) * met(1, i - 2, j, k) *
///                        (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
///                         c1 * (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
///                        stry(j) +
///                    la(i - 2, j, k) * met(4, i - 2, j, k) * met(1, i - 2, j, k) *
///                        (c2 * (u(3, i - 2, j, k + 2) - u(3, i - 2, j, k - 2)) +
///                         c1 *
///                             (u(3, i - 2, j, k + 1) - u(3, i - 2, j, k - 1))))) +
///              c1 *
///                  ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
///                       met(2, i + 1, j, k) * met(1, i + 1, j, k) *
///                       (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
///                        c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) *
///                       strx(i + 1) +
///                   la(i + 1, j, k) * met(3, i + 1, j, k) * met(1, i + 1, j, k) *
///                       (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
///                        c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
///                       stry(j) +
///                   la(i + 1, j, k) * met(4, i + 1, j, k) * met(1, i + 1, j, k) *
///                       (c2 * (u(3, i + 1, j, k + 2) - u(3, i + 1, j, k - 2)) +
///                        c1 * (u(3, i + 1, j, k + 1) - u(3, i + 1, j, k - 1))) -
///                   ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
///                        met(2, i - 1, j, k) * met(1, i - 1, j, k) *
///                        (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
///                         c1 * (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) *
///                        strx(i - 1) +
///                    la(i - 1, j, k) * met(3, i - 1, j, k) * met(1, i - 1, j, k) *
///                        (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
///                         c1 * (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
///                        stry(j) +
///                    la(i - 1, j, k) * met(4, i - 1, j, k) * met(1, i - 1, j, k) *
///                        (c2 * (u(3, i - 1, j, k + 2) - u(3, i - 1, j, k - 2)) +
///                         c1 * (u(3, i - 1, j, k + 1) -
///                               u(3, i - 1, j, k - 1)))))) *
///             istry;
///
///         // qr derivatives
///         // 82 ops, tot=687
///         r1 +=
///             c2 *
///                 (mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
///                      (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
///                       c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) *
///                      stry(j) * istrx +
///                  la(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
///                      (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
///                       c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) -
///                  (mu(i, j, k - 2) * met(3, i, j, k - 2) * met(1, i, j, k - 2) *
///                       (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
///                        c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *
///                       stry(j) * istrx +
///                   la(i, j, k - 2) * met(2, i, j, k - 2) * met(1, i, j, k - 2) *
///                       (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
///                        c1 * (u(2, i, j + 1, k - 2) - u(2, i, j - 1, k - 2))))) +
///             c1 * (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
///                       (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
///                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *
///                       stry(j) * istrx +
///                   la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
///                       (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
///                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) -
///                   (mu(i, j, k - 1) * met(3, i, j, k - 1) * met(1, i, j, k - 1) *
///                        (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
///                         c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *
///                        stry(j) * istrx +
///                    la(i, j, k - 1) * met(2, i, j, k - 1) * met(1, i, j, k - 1) *
///                        (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
///                         c1 * (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1)))));
///
///         // rq derivatives
///         // 82 ops, tot=769
///         r1 +=
///             c2 *
///                 (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *
///                      (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
///                       c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *
///                      stry(j + 2) * istrx +
///                  mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
///                      (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
///                       c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) -
///                  (mu(i, j - 2, k) * met(3, i, j - 2, k) * met(1, i, j - 2, k) *
///                       (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
///                        c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *
///                       stry(j - 2) * istrx +
///                   mu(i, j - 2, k) * met(2, i, j - 2, k) * met(1, i, j - 2, k) *
///                       (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
///                        c1 * (u(2, i, j - 2, k + 1) - u(2, i, j - 2, k - 1))))) +
///             c1 * (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *
///                       (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
///                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *
///                       stry(j + 1) * istrx +
///                   mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
///                       (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
///                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) -
///                   (mu(i, j - 1, k) * met(3, i, j - 1, k) * met(1, i, j - 1, k) *
///                        (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
///                         c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *
///                        stry(j - 1) * istrx +
///                    mu(i, j - 1, k) * met(2, i, j - 1, k) * met(1, i, j - 1, k) *
///                        (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
///                         c1 * (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1)))));
///
///         // 4 ops, tot=773
///         lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;
///       }

#ifndef RAJAPerf_Apps_SW4CK_KERNEL_2_HPP
#define RAJAPerf_Apps_SW4CK_KERNEL_2_HPP

using float_sw4 = double;

#define SW4CK_KERNEL_2_DATA_SETUP                           \
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
  const int istart = m_istart;                              \
  const int ifirst = m_first;                               \
  const int ilast = m_ilast;                                \
  const int jstart = m_jstart;                              \
  const int jfirst = m_jfirst;                              \
  const int jend = m_jend;                                  \
  const int jlast = m_jlast;                                \
  const int kfirst = m_kfirst;                              \
  const int kstart = m_kstart;                              \
  const int klast = m_klast;                                \
  const int kend = m_kend;                                  \
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
  Real_ptr a_mu = m_a_mu;                                   \
  Real_ptr a_lambda = m_a_lambda;                           \
  Real_ptr a_jac = m_a_jac;                                 \
  Real_ptr a_u = m_a_u;                                     \
  Real_ptr a_lu = m_a_lu;                                   \
  Real_ptr a_met = m_a_met;                                 \
  Real_ptr a_strx = m_a_strx;                               \
  Real_ptr a_stry = m_a_stry;
/*
  Real_ptr a_acof = m_a_acof;                                  \
  Real_ptr a_bope = m_a_bope;                               \
  Real_ptr a_ghcof = m_a_ghcof;                             \
  Real_ptr a_acof_no_gp = m_a_acof_no_gp;      \
  Real_ptr a_ghcof_no_gp = m_a_ghcof_no_gp;
*/

// Direct reuse of fortran code by these macro definitions:
#define mu(i, j, k) a_mu[base + (i) + ni * (j) + nij * (k)]
#define la(i, j, k) a_lambda[base + (i) + ni * (j) + nij * (k)]
#define jac(i, j, k) a_jac[base + (i) + ni * (j) + nij * (k)]
#define u(c, i, j, k) a_u[base3 + (i) + ni * (j) + nij * (k) + nijk * (c)]
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
#define SW4CK_KERNEL_2_BODY_1                                \
  float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);         \
          float_sw4 istry = 1 / (stry(j));                   \
          float_sw4 istrx = 1 / (strx(i));      \
          float_sw4 istrxy = istry * istrx;     \
                                                \
          float_sw4 r1 = 0;

// pp derivative (u)
// 53 ops, tot=58
#define SW4CK_KERNEL_2_BODY_2                                           \
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
  float_sw4 mux1 = cof2 - tf * (cof3 + cof1);                           \
  float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                     \
  float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                     \
  float_sw4 mux4 = cof4 - tf * (cof3 + cof5);                           \
                                                                        \
  r1 += i6 *                                                            \
    (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +                       \
     mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +                       \
     mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +                       \
     mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *                      \
    istry;

// qq derivative (u)
// 43 ops, tot=101
#define SW4CK_KERNEL_2_BODY_3                                           \
  {                                                                     \
    float_sw4 cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *          \
      met(1, i, j - 2, k) * stry(j - 2);                                \
    float_sw4 cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *          \
      met(1, i, j - 1, k) * stry(j - 1);                                \
    float_sw4 cof3 =                                                    \
      (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);      \
    float_sw4 cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *          \
      met(1, i, j + 1, k) * stry(j + 1);                                \
    float_sw4 cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *          \
      met(1, i, j + 2, k) * stry(j + 2);                                \
    float_sw4 mux1 = cof2 - tf * (cof3 + cof1);                         \
    float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                   \
    float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                   \
    float_sw4 mux4 = cof4 - tf * (cof3 + cof5);                         \
                                                                        \
    r1 += i6 *                                                          \
      (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +                     \
       mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +                     \
       mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +                     \
       mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *                    \
      istrx;                                                            \
  }
// rr derivative (u)
// 5*11+14+14=83 ops, tot=184
#define SW4CK_KERNEL_2_BODY_4                                           \
  {                                                                     \
  float_sw4 cof1 =                                                      \
    (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *     \
    strx(i) * met(2, i, j, k - 2) * strx(i) +                           \
    mu(i, j, k - 2) * (met(3, i, j, k - 2) * stry(j) *                  \
                       met(3, i, j, k - 2) * stry(j) +                  \
                       met(4, i, j, k - 2) * met(4, i, j, k - 2));      \
  float_sw4 cof2 =                                                      \
    (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *     \
    strx(i) * met(2, i, j, k - 1) * strx(i) +                           \
    mu(i, j, k - 1) * (met(3, i, j, k - 1) * stry(j) *                  \
                       met(3, i, j, k - 1) * stry(j) +                  \
                       met(4, i, j, k - 1) * met(4, i, j, k - 1));      \
  float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) *  \
    strx(i) * met(2, i, j, k) * strx(i) +                               \
    mu(i, j, k) * (met(3, i, j, k) * stry(j) *                          \
                   met(3, i, j, k) * stry(j) +                          \
                   met(4, i, j, k) * met(4, i, j, k));                  \
  float_sw4 cof4 =                                                      \
    (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *     \
    strx(i) * met(2, i, j, k + 1) * strx(i) +                           \
    mu(i, j, k + 1) * (met(3, i, j, k + 1) * stry(j) *                  \
                       met(3, i, j, k + 1) * stry(j) +                  \
                       met(4, i, j, k + 1) * met(4, i, j, k + 1));      \
  float_sw4 cof5 =                                                      \
    (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *     \
    strx(i) * met(2, i, j, k + 2) * strx(i) +                           \
    mu(i, j, k + 2) * (met(3, i, j, k + 2) * stry(j) *                  \
                       met(3, i, j, k + 2) * stry(j) +                  \
                       met(4, i, j, k + 2) * met(4, i, j, k + 2));      \
                                                                        \
  float_sw4 mux1 = cof2 - tf * (cof3 + cof1);                           \
  float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                     \
  float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                     \
  float_sw4 mux4 = cof4 - tf * (cof3 + cof5);                           \
                                                                        \
  r1 += i6 *                                                            \
    (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +                       \
     mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +                       \
     mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +                       \
     mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *                      \
    istrxy;                                                             \
  }

// rr derivative (v)
// 42 ops, tot=226
#define SW4CK_KERNEL_2_BODY_5                                           \
  cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *    \
    met(3, i, j, k - 2);                                                \
  cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *    \
    met(3, i, j, k - 1);                                                \
  cof3 =                                                                \
    (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);    \
  cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *    \
    met(3, i, j, k + 1);                                                \
  cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *    \
    met(3, i, j, k + 2);                                                \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r1 += i6 * (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +              \
              mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +              \
              mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +              \
              mux4 * (u(2, i, j, k + 2) - u(2, i, j, k)));

// rr derivative (w)
// 43 ops, tot=269
#define SW4CK_KERNEL_2_BODY_6                                           \
  cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *    \
    met(4, i, j, k - 2);                                                \
  cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *    \
    met(4, i, j, k - 1);                                                \
  cof3 =                                                                \
    (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);    \
  cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *    \
    met(4, i, j, k + 1);                                                \
  cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *    \
    met(4, i, j, k + 2);                                                \
  mux1 = cof2 - tf * (cof3 + cof1);                                     \
  mux2 = cof1 + cof4 + 3 * (cof3 + cof2);                               \
  mux3 = cof2 + cof5 + 3 * (cof4 + cof3);                               \
  mux4 = cof4 - tf * (cof3 + cof5);                                     \
                                                                        \
  r1 += i6 *                                                            \
    (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +                       \
     mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +                       \
     mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +                       \
     mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *                      \
    istry;

// pq-derivatives
// 38 ops, tot=307
#define SW4CK_KERNEL_2_BODY_7                                           \
  r1 +=                                                                 \
    c2 *                                                                \
    (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *      \
    (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +             \
     c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -            \
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

// qp-derivatives
// 38 ops, tot=345
#define SW4CK_KERNEL_2_BODY_8                                           \
  r1 +=                                                                 \
    c2 *                                                                \
    (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *      \
    (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +             \
     c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -            \
     la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *      \
     (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +            \
      c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +          \
    c1 *                                                                \
    (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *      \
     (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +            \
      c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -           \
     la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *      \
     (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +            \
      c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

// pr-derivatives
// 130 ops., tot=475
#define SW4CK_KERNEL_2_BODY_9                                           \
  r1 +=                                                                 \
    c2 *                                                                \
    ((2 * mu(i, j, k + 2) + la(i, j, k + 2)) *                          \
     met(2, i, j, k + 2) * met(1, i, j, k + 2) *                        \
     (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +            \
      c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *           \
     strx(i) * istry +                                                  \
     mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *      \
     (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +            \
      c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) +           \
     mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *      \
     (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +            \
      c1 * (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *           \
     istry -                                                            \
     ((2 * mu(i, j, k - 2) + la(i, j, k - 2)) *                         \
      met(2, i, j, k - 2) * met(1, i, j, k - 2) *                       \
      (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +           \
       c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) *          \
      strx(i) * istry +                                                 \
      mu(i, j, k - 2) * met(3, i, j, k - 2) *                           \
      met(1, i, j, k - 2) *                                             \
      (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +           \
       c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) +          \
      mu(i, j, k - 2) * met(4, i, j, k - 2) *                           \
      met(1, i, j, k - 2) *                                             \
      (c2 * (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +           \
       c1 * (u(3, i + 1, j, k - 2) - u(3, i - 1, j, k - 2))) *          \
      istry)) +                                                         \
    c1 *                                                                \
    ((2 * mu(i, j, k + 1) + la(i, j, k + 1)) *                          \
     met(2, i, j, k + 1) * met(1, i, j, k + 1) *                        \
     (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +            \
      c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *           \
     strx(i) * istry +                                                  \
     mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *      \
     (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +            \
      c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) +           \
     mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *      \
     (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +            \
      c1 * (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *           \
     istry -                                                            \
     ((2 * mu(i, j, k - 1) + la(i, j, k - 1)) *                         \
      met(2, i, j, k - 1) * met(1, i, j, k - 1) *                       \
      (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +           \
       c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) *          \
      strx(i) * istry +                                                 \
      mu(i, j, k - 1) * met(3, i, j, k - 1) *                           \
      met(1, i, j, k - 1) *                                             \
      (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +           \
       c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) +          \
      mu(i, j, k - 1) * met(4, i, j, k - 1) *                           \
      met(1, i, j, k - 1) *                                             \
      (c2 * (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +           \
       c1 * (u(3, i + 1, j, k - 1) - u(3, i - 1, j, k - 1))) *          \
      istry));

// rp derivatives
// 130 ops, tot=605
#define SW4CK_KERNEL_2_BODY_10                                          \
  r1 +=                                                                 \
    (c2 *                                                               \
    ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *                          \
    met(2, i + 2, j, k) * met(1, i + 2, j, k) *                         \
     (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +            \
      c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) *           \
     strx(i + 2) +                                                      \
     la(i + 2, j, k) * met(3, i + 2, j, k) *                            \
     met(1, i + 2, j, k) *                                              \
     (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +            \
      c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *           \
     stry(j) +                                                          \
     la(i + 2, j, k) * met(4, i + 2, j, k) *                            \
     met(1, i + 2, j, k) *                                              \
     (c2 * (u(3, i + 2, j, k + 2) - u(3, i + 2, j, k - 2)) +            \
      c1 * (u(3, i + 2, j, k + 1) - u(3, i + 2, j, k - 1))) -           \
     ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *                         \
      met(2, i - 2, j, k) * met(1, i - 2, j, k) *                       \
      (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +           \
       c1 *                                                             \
       (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) *               \
      strx(i - 2) +                                                     \
      la(i - 2, j, k) * met(3, i - 2, j, k) *                           \
      met(1, i - 2, j, k) *                                             \
      (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +           \
       c1 *                                                             \
       (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *               \
      stry(j) +                                                         \
      la(i - 2, j, k) * met(4, i - 2, j, k) *                           \
      met(1, i - 2, j, k) *                                             \
      (c2 * (u(3, i - 2, j, k + 2) - u(3, i - 2, j, k - 2)) +           \
       c1 * (u(3, i - 2, j, k + 1) -                                    \
             u(3, i - 2, j, k - 1))))) +                                \
     c1 *                                                               \
     ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *                         \
                    met(2, i + 1, j, k) * met(1, i + 1, j, k) *         \
      (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +           \
       c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) *          \
      strx(i + 1) +                                                     \
      la(i + 1, j, k) * met(3, i + 1, j, k) *                           \
      met(1, i + 1, j, k) *                                             \
      (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +           \
       c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *          \
      stry(j) +                                                         \
      la(i + 1, j, k) * met(4, i + 1, j, k) *                           \
      met(1, i + 1, j, k) *                                             \
      (c2 * (u(3, i + 1, j, k + 2) - u(3, i + 1, j, k - 2)) +           \
       c1 * (u(3, i + 1, j, k + 1) - u(3, i + 1, j, k - 1))) -          \
      ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *                        \
       met(2, i - 1, j, k) * met(1, i - 1, j, k) *                      \
       (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +          \
        c1 *                                                            \
        (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) *              \
       strx(i - 1) +                                                    \
       la(i - 1, j, k) * met(3, i - 1, j, k) *                          \
       met(1, i - 1, j, k) *                                            \
       (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +          \
        c1 *                                                            \
        (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *              \
       stry(j) +                                                        \
       la(i - 1, j, k) * met(4, i - 1, j, k) *                          \
       met(1, i - 1, j, k) *                                            \
       (c2 * (u(3, i - 1, j, k + 2) - u(3, i - 1, j, k - 2)) +          \
        c1 * (u(3, i - 1, j, k + 1) -                                   \
              u(3, i - 1, j, k - 1)))))) *                              \
    istry;

// qr derivatives
// 82 ops, tot=687
#define SW4CK_KERNEL_2_BODY_11                                          \
  r1 +=                                                                 \
    c2 *                                                                \
    (mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *      \
     (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +            \
      c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) *           \
     stry(j) * istrx +                                                  \
     la(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *      \
     (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +            \
      c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) -           \
     (mu(i, j, k - 2) * met(3, i, j, k - 2) *                           \
      met(1, i, j, k - 2) *                                             \
      (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +           \
       c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *          \
      stry(j) * istrx +                                                 \
      la(i, j, k - 2) * met(2, i, j, k - 2) *                           \
      met(1, i, j, k - 2) *                                             \
      (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +           \
       c1 * (u(2, i, j + 1, k - 2) -                                    \
             u(2, i, j - 1, k - 2))))) +                                \
    c1 *                                                                \
    (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *      \
     (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +            \
      c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *           \
     stry(j) * istrx +                                                  \
     la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *      \
     (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +            \
      c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) -           \
     (mu(i, j, k - 1) * met(3, i, j, k - 1) *                           \
                        met(1, i, j, k - 1) * \
      (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +           \
       c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *          \
      stry(j) * istrx +                                                 \
      la(i, j, k - 1) * met(2, i, j, k - 1) *                           \
      met(1, i, j, k - 1) *                                             \
      (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +           \
       c1 *                                                             \
       (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1)))));

// rq derivatives
// 82 ops, tot=769
#define SW4CK_KERNEL_2_BODY_12                                          \
  r1 +=                                                                 \
    c2 *                                                                \
    (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *      \
     (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +            \
      c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *           \
     stry(j + 2) * istrx +                                              \
     mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *      \
     (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +            \
      c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) -           \
     (mu(i, j - 2, k) * met(3, i, j - 2, k) *                           \
      met(1, i, j - 2, k) *                                             \
      (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +           \
       c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *          \
      stry(j - 2) * istrx +                                             \
      mu(i, j - 2, k) * met(2, i, j - 2, k) *                           \
      met(1, i, j - 2, k) *                                             \
      (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +           \
       c1 * (u(2, i, j - 2, k + 1) -                                    \
             u(2, i, j - 2, k - 1))))) +                                \
    c1 *                                                                \
    (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *      \
     (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +            \
      c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *           \
     stry(j + 1) * istrx +                                              \
     mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *      \
     (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +            \
      c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) -           \
     (mu(i, j - 1, k) * met(3, i, j - 1, k) *                           \
      met(1, i, j - 1, k) *                                             \
      (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +           \
       c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *          \
      stry(j - 1) * istrx +                                             \
      mu(i, j - 1, k) * met(2, i, j - 1, k) *                           \
      met(1, i, j - 1, k) *                                             \
      (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +           \
       c1 *                                                             \
       (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1)))));

// 4 ops, tot=773
#define SW4CK_KERNEL_2_BODY_13 \
  lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;


#include "common/KernelBase.hpp"

  namespace rajaperf {
  class RunParams;

  namespace apps {

  class SW4CK_KERNEL_2 : public KernelBase {
  public:
    SW4CK_KERNEL_2(const RunParams &params);

    ~SW4CK_KERNEL_2();

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

    const int m_istart;
    const int m_first;
    const int m_ilast;
    const int m_jstart;
    const int m_jfirst;
    const int m_jend;
    const int m_jlast;
    const int m_kfirst;
    const int m_kstart;
    const int m_klast;
    const int m_kend;

  };

  } // end namespace apps
  } // end namespace rajaperf

#endif // closing endif for header file include guard
