#!/usr/bin/env python3

import math
import os
import sys
import re
import getopt
import csv
import matplotlib.pyplot as plt


def normalize_color_tuple(t):
   len_t = 0.0
   for i in range(0, len(t)):
      len_t += t[i]*t[i]
   len_t = math.sqrt(len_t)
   new_t = ()
   for i in range(0, len(t)):
      new_t += (t[i]/len_t,)
   return new_t

def clamp_tuple(t, min_val=0.0, max_val=1.0):
   new_t = ()
   for i in range(0, len(t)):
      val = t[i]
      if val > max_val:
         val = max_val
      elif val < min_val:
         val = min_val
      new_t += (val,)
   return new_t

def color_mul(t, factor):
   new_t = ()
   for i in range(0, len(t)):
      new_t += (t[i]*factor,)
   return clamp_tuple(new_t)

def make_color_tuple_rgb(r, g, b):
   return (r/255.0, g/255.0, b/255.0)

def make_color_tuple_str(color_str):
   color_str = color_str.strip()
   if len(color_str) < 2 or color_str[0] != "(" or color_str[len(color_str)-1] != ")":
      raise NameError("Expected a tuple of 3 floats in [0-1]")
   color_str = color_str[1:len(color_str)-1]
   rgb = color_str.split(",")
   if len(rgb) != 3:
      raise NameError("Expected a tuple of 3 floats in [0-1]")
   r = float(rgb[0].strip())
   g = float(rgb[1].strip())
   b = float(rgb[2].strip())
   return clamp_tuple((r, g, b))


g_color_base_factor = 1.0
g_color_lambda_factor = 0.7
g_color_raja_factor = 0.4
g_color_seq = normalize_color_tuple(make_color_tuple_rgb(204, 119, 34)) # ocre
g_color_omp = normalize_color_tuple(make_color_tuple_rgb(0, 115, 125)) # omp teal
g_color_ompt = normalize_color_tuple(make_color_tuple_rgb(125, 10, 0)) # omp teal compliment
g_color_cuda = normalize_color_tuple(make_color_tuple_rgb(118, 185, 0)) # nvidia green
g_color_hip = normalize_color_tuple(make_color_tuple_rgb(237, 28, 36)) # amd red
g_known_variants = { "Base_Seq": {"color": color_mul(g_color_seq, g_color_base_factor)},
                     "Lambda_Seq": {"color": color_mul(g_color_seq, g_color_lambda_factor)},
                     "RAJA_Seq": {"color": color_mul(g_color_seq, g_color_raja_factor)},

                     "Base_OpenMP": {"color": color_mul(g_color_omp, g_color_base_factor)},
                     "Lambda_OpenMP": {"color": color_mul(g_color_omp, g_color_lambda_factor)},
                     "RAJA_OpenMP": {"color": color_mul(g_color_omp, g_color_raja_factor)},

                     "Base_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_base_factor)},
                     "Lambda_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_lambda_factor)},
                     "RAJA_OpenMPTarget": {"color": color_mul(g_color_ompt, g_color_raja_factor)},

                     "Base_CUDA": {"color": color_mul(g_color_cuda, g_color_base_factor)},
                     "Lambda_CUDA": {"color": color_mul(g_color_cuda, g_color_lambda_factor)},
                     "RAJA_CUDA": {"color": color_mul(g_color_cuda, g_color_raja_factor)},

                     "Base_HIP": {"color": color_mul(g_color_hip, g_color_base_factor)},
                     "Lambda_HIP": {"color": color_mul(g_color_hip, g_color_lambda_factor)},
                     "RAJA_HIP": {"color": color_mul(g_color_hip, g_color_raja_factor)}
                   }
g_known_tunings =  { "default": {"format": "-"},
                     "block_25": {"format": "-"},
                     "block_32": {"format": ":"},
                     "block_64": {"format": "-."},
                     "block_128": {"format": "--"},
                     "block_256": {"format": "-"},
                     "block_512": {"format": "-."},
                     "block_1024": {"format": "-"},
                     "cub": {"format": ":"},
                     "rocprim": {"format": ":"}
                   }
g_markers = [ "o", "s", "+", "x", "*", "d", "h", "p", "8" ]

# reformat or color series
# formatted as series_name: dictionary of "color": color, "format": format
g_series_reformat = {}

g_timing_filename = "RAJAPerf-timing-Minimum.csv"
g_runinfo_filename = "RAJAPerf-kernels.csv"
g_timing_file_kind = "time(s)"

# Kernels sorted into categories based on performance bottlenecks

g_known_kernel_groups = {
   "bandwidth": {
      "kind": "bandwidth(GiB/s)",
      "kernels": [ "Basic_DAXPY", "Basic_DAXPY_ATOMIC", "Basic_INIT3",
                   "Basic_INIT_VIEW1D", "Basic_INIT_VIEW1D_OFFSET",
                   "Basic_MULADDSUB", "Lcals_DIFF_PREDICT", "Lcals_EOS",
                   "Lcals_FIRST_DIFF", "Lcals_FIRST_SUM", "Lcals_GEN_LIN_RECUR",
                   "Lcals_HYDRO_1D", "Lcals_PLANCKIAN", "Lcals_TRIDIAG_ELIM",
                   "Polybench_JACOBI_1D", "Stream_ADD", "Stream_COPY",
                   "Stream_MUL", "Stream_TRIAD",

                   "Basic_IF_QUAD", "Basic_INDEXLIST", "Basic_INDEXLIST_3LOOP",
                   "Basic_NESTED_INIT", "Lcals_HYDRO_2D", "Lcals_INT_PREDICT",
                   "Polybench_FDTD_2D", "Polybench_HEAT_3D",
                   "Polybench_JACOBI_2D", "Stream_DOT", "Apps_CONVECTION3DPA",
                   "Apps_DEL_DOT_VEC_2D", "Apps_DIFFUSION3DPA", "Apps_ENERGY",
                   "Apps_FIR", "Apps_MASS3DPA", "Apps_NODAL_ACCUMULATION_3D",
                   "Apps_PRESSURE", "Apps_VOL3D", "Algorithm_SCAN",
                   "Algorithm_REDUCE_SUM", ],
   },
   "flops": {
      "kind": "GFLOPS",
      "kernels": [ "Basic_MAT_MAT_SHARED", "Polybench_2MM", "Polybench_3MM",
                   "Polybench_GEMM",

                   "Polybench_HEAT_3D", "Apps_CONVECTION3DPA",
                   "Apps_DEL_DOT_VEC_2D", "Apps_DIFFUSION3DPA", "Apps_FIR",
                   "Apps_MASS3DPA", "Apps_VOL3D", ],
   },
   "reduce": {
      "kind": "throughput(GProblem size/s)",
      "kernels": [ "Basic_PI_REDUCE", "Basic_REDUCE3_INT", "Basic_REDUCE_STRUCT",
                   "Basic_TRAP_INT", "Lcals_FIRST_MIN", "Stream_DOT",
                   "Algorithm_REDUCE_SUM", ]
   },
   "other": {
      "kind": "throughput(GProblem size/s)",
      "kernels": [ "Polybench_ADI", "Polybench_ATAX", "Polybench_FLOYD_WARSHALL",
                   "Polybench_GEMVER", "Polybench_GESUMMV", "Polybench_MVT",
                   "Apps_LTIMES", "Apps_LTIMES_NOVIEW", "Algorithm_SORT",
                   "Algorithm_SORTPAIRS", ]
   },
   "launch_bound": {
      "kind": "time/rep(us)",
      "kernels": [ "Apps_HALOEXCHANGE", "Apps_HALOEXCHANGE_FUSED", ]
   },
   }

def first(vals):
   return vals[0]

def last(vals):
   return vals[len(vals)-1]

def sum(vals):
   sum_val = 0
   for val in vals:
      sum_val += val
   return sum_val

def avg(vals):
   return sum(vals) / len(vals)

def stddev(vals):
   avg_val = avg(vals)
   stddev_val = 0
   for val in vals:
      stddev_val += (val - avg_val)*(val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val

def relstddev(vals):
   avg_val = avg(vals)
   stddev_val = 0
   for val in vals:
      stddev_val += (val - avg_val)*(val - avg_val)
   stddev_val /= len(vals)
   stddev_val = math.sqrt(stddev_val)
   return stddev_val / abs(avg_val)

# returns (intercept, slope, correlation_coefficient)
def linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum):
   assert(n>0)
   if n == 1:
      slope = 0.0
      intercept = ysum
      correlation_coefficient = 1.0
   else:
      slope = (n*xysum - xsum*ysum) / ((n*x2sum - xsum*xsum)+1e-80)
      intercept = (ysum - slope*xsum)/n
      correlation_coefficient = (n*xysum - xsum*ysum) / (math.sqrt((n*x2sum - xsum*xsum)*(n*y2sum - ysum*ysum))+1e-80)
   return (intercept, slope, correlation_coefficient)

# returns (intercept, slope, correlation_coefficient)
def linearRegression(yvals, xvals):
   assert(len(xvals) == len(yvals))
   n = len(xvals)
   xsum = sum(xvals)
   ysum = sum(yvals)
   x2sum = sum([x*x for x in xvals])
   y2sum = sum([y*y for y in yvals])
   xysum = sum([xvals[i]*yvals[i] for i in range(0, n)])
   return linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum)

def eval_linearRegression(lr_vals, xval):
   return lr_vals[0] + lr_vals[1]*xval

# returns (intercept, slope, correlation_coefficient)
def linearRegression_loglog(yvals, xvals):
   assert(len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return linearRegression(ylogvals, xlogvals)

def eval_linearRegression_loglog(lr_vals, xval):
   return math.pow(2, lr_vals[0])*math.pow(xval, lr_vals[1])


# returns (intercept, slope, correlation_coefficient)
def segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR):
   sums[0] += xvals[i]
   sums[1] += yvals[i]
   sums[2] += xvals[i]*xvals[i]
   sums[3] += yvals[i]*yvals[i]
   sums[4] += xvals[i]*yvals[i]
   xsum = sums[0]
   ysum = sums[1]
   x2sum = sums[2]
   y2sum = sums[3]
   xysum = sums[4]
   LR[i] = linearRegression_helper(n, xsum, ysum, x2sum, y2sum, xysum)

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression_helper(ret, i, n, xvals, yvals, denom, LR_left, LR_right):
   lr_vals_left = None
   lr_vals_right = None
   break_point = None

   if i == 0:
      lr_vals_right = LR_right[i]
      break_point = xvals[i] - 1.0
   elif i > 0 and i < n:
      lr_vals_left = LR_left[i-1]
      lr_vals_right = LR_right[i]
      break_point = (xvals[i-1] + xvals[i]) / 2.0
   elif i == n:
      lr_vals_left = LR_left[i-1]
      break_point = xvals[i-1] + 1.0
   else:
      assert(0)

   numer = 0.0
   for j in range(0, n):
      xval = xvals[j]
      yval = yvals[j]
      lr_vals = None
      if xval < break_point:
         lr_vals = lr_vals_left
      else:
         lr_vals = lr_vals_right
      lr_yval = eval_linearRegression(lr_vals, xval)
      numer += (yval - lr_yval)*(yval - lr_yval)

   correlation_coefficient = 1.0 - numer / denom
   if correlation_coefficient > ret[2]:
      ret[0] = [break_point,]
      ret[1] = [lr_vals_left, lr_vals_right,]
      ret[2] = correlation_coefficient

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression(yvals, xvals):
   assert(len(xvals) == len(yvals))
   N = len(xvals)

   LR_left = []
   LR_right = []
   for i in range(0, N):
      LR_left.append(None)
      LR_right.append(None)

   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for ii in range(0, N):
      i = N-ii-1
      n = ii+1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_right)

   sums = [0.0, 0.0, 0.0, 0.0, 0.0]
   for i in range(0, N):
      n = i+1
      segmented_linearRegression_partialRegression(i, n, xvals, yvals, sums, LR_left)

   yavg = avg(yvals)
   denom = sum([(y-yavg)*(y-yavg) for y in yvals])
   ret = [[], [], -math.inf]
   for i in range(0, N+1):
      segmented_linearRegression_helper(ret, i, N, xvals, yvals, denom, LR_left, LR_right)

   return (*ret,)

def find_segment(break_points, xval):
   break_i = len(break_points)
   for i in range(0, len(break_points)):
      break_point = break_points[i]
      if xval < break_point:
         break_i = i
         break
   return break_i

def eval_segmented_linearRegression(slr_vals, xval):
   break_i = find_segment(slr_vals[0], xval)
   return eval_linearRegression(slr_vals[1][break_i], xval)

# returns ([break points...], [linear regressions...], correlation_coefficient)
def segmented_linearRegression_loglog(yvals, xvals):
   assert(len(xvals) == len(yvals))
   xlogvals = [math.log(x, 2) for x in xvals]
   ylogvals = [math.log(y, 2) for y in yvals]
   return segmented_linearRegression(ylogvals, xlogvals)

def eval_segmented_linearRegression_loglog(slr_vals, xval):
   break_i = find_segment(slr_vals[0], math.log(xval, 2))
   return eval_linearRegression_loglog(slr_vals[1][break_i], xval)


class Data:

   num_sweeps = 0
   sweeps = {}
   sweep_markers = {}
   exclude_sweeps = {}

   num_run_sizes = 0
   run_sizes = {}

   include_kernel_groups = {}
   exclude_kernel_groups = {}

   num_kernels = 0
   kernels = {}
   include_kernels = {}
   exclude_kernels = {}

   num_variants = 0
   variants = {}
   variant_colors = {}
   include_variants = {}
   exclude_variants = {}

   num_tunings = 0
   tunings = {}
   tuning_formats = {}
   include_tunings = {}
   exclude_tunings = {}

   def add_sweep(sweep_name):
      sweep_index = Data.num_sweeps
      Data.num_sweeps += 1
      Data.sweeps[sweep_name] = sweep_index
      Data.sweeps[sweep_index] = sweep_name

   def add_run_size(run_size_name):
      run_size_index = Data.num_run_sizes
      Data.num_run_sizes += 1
      Data.run_sizes[run_size_name] = run_size_index
      Data.run_sizes[run_size_index] = run_size_name

   def add_kernel(kernel_name):
      kernel_index = Data.num_kernels
      Data.num_kernels += 1
      Data.kernels[kernel_name] = kernel_index
      Data.kernels[kernel_index] = kernel_name

   def add_variant(variant_name):
      variant_index = Data.num_variants
      Data.num_variants += 1
      Data.variants[variant_name]  = variant_index
      Data.variants[variant_index] = variant_name
      if variant_name in g_known_variants:
         variant_color = g_known_variants[variant_name]["color"]
         Data.variant_colors[variant_name] = variant_color
         Data.variant_colors[variant_index] = variant_color
      else:
         print("Unknown variant {0}".format(variant_name))
         sys.exit(1)

   def add_tuning(tuning_name):
      tuning_index = Data.num_tunings
      Data.num_tunings += 1
      Data.tunings[tuning_name]  = tuning_index
      Data.tunings[tuning_index] = tuning_name
      if tuning_name in g_known_tunings:
         tuning_format = g_known_tunings[tuning_name]["format"]
         Data.tuning_formats[tuning_name] = tuning_format
         Data.tuning_formats[tuning_index] = tuning_format
      else:
         print("Unknown tuning {0}".format(tuning_name))
         sys.exit(1)

   num_axes = 5
   axes = { "sweep_dir_name": 0, 0: "sweep_dir_name",
            "run_size": 1,       1: "run_size",
            "kernel_index": 2,   2: "kernel_index",
            "variant_index": 3,  3: "variant_index",
            "tuning_index": 4,   4: "tuning_index", }

   def get_axis_name(axis_index):
      if axis_index in Data.axes:
         return Data.axes[axis_index]
      else:
         raise NameError("Unknown axis index {}".format(axis_index))

   def get_index_name(axis_index, index):
      if axis_index == Data.axes["sweep_dir_name"]:
         return Data.sweeps[index]
      elif axis_index == Data.axes["run_size"]:
         return Data.run_sizes[index]
      elif axis_index == Data.axes["kernel_index"]:
         return Data.kernels[index]
      elif axis_index == Data.axes["variant_index"]:
         return Data.variants[index]
      elif axis_index == Data.axes["tuning_index"]:
         return Data.tunings[index]
      else:
         raise NameError("Unknown axis index {}".format(axis_index))

   def get_axis_index_str(axis_index, index):
      return "{}:{}".format(Data.get_axis_name(axis_index), Data.get_index_name(axis_index, index))

   def get_axes_index_str(axes_index):
      name = "{"
      for axis_index, index in axes_index.items():
         name = "{}{},".format(name, Data.get_axis_index_str(axis_index, index))
      return "{}}}".format(name)

   def get_axis_index(axis_name, index_name):
      if axis_name == "sweep_dir_name":
         return {Data.axes[axis_name]: Data.sweeps[index_name],}
      elif axis_name == "run_size":
         return {Data.axes[axis_name]: Data.run_sizes[index_name],}
      elif axis_name == "kernel_index":
         return {Data.axes[axis_name]: Data.kernels[index_name],}
      elif axis_name == "variant_index":
         return {Data.axes[axis_name]: Data.variants[index_name],}
      elif axis_name == "tuning_index":
         return {Data.axes[axis_name]: Data.tunings[index_name],}
      else:
         raise NameError("Unknown axis name {}".format(axis_name))

   def axes_difference(axes, partial_axes_index):
      new_axes = []
      for axis_index in axes:
         if not axis_index in partial_axes_index:
            new_axes.append(axis_index)
      return new_axes

   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   info_axes = [ axes["sweep_dir_name"],
                 axes["run_size"],
                 axes["kernel_index"], ]

   # multi-dimensional array structured like this
   #     directory name - platform, compiler, etc
   #       run size - problem size, for run_sizes
   #         kernel index - for kernels
   #           variant index - for variants
   #             tuning index - for tunings
   data_axes = [ axes["sweep_dir_name"],
                 axes["run_size"],
                 axes["kernel_index"],
                 axes["variant_index"],
                 axes["tuning_index"], ]

   # multi-dimensional array structured like data but missing some dimensions
   #     directory name - platform, compiler, etc
   #       kernel index - for kernels
   #         variant index - for variants
   #           tuning index - for tunings
   run_size_reduced_axes = [ axes["sweep_dir_name"],
                             axes["kernel_index"],
                             axes["variant_index"],
                             axes["tuning_index"], ]

   data_model_kind = "time(s)"



   def MultiAxesTreeKeyGenerator0(data_tree):
      assert(len(data_tree.axes) == 0)
      if False:
         yield {}

   def MultiAxesTreeKeyGenerator1(data_tree):
      assert(len(data_tree.axes) == 1)
      assert(data_tree.data)
      for k0 in data_tree.data.keys():
         yield {data_tree.axes[0]: k0,}

   def MultiAxesTreeKeyGenerator2(data_tree):
      assert(len(data_tree.axes) == 2)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1 in v0.keys():
            yield {data_tree.axes[0]: k0,
                   data_tree.axes[1]: k1,}

   def MultiAxesTreeKeyGenerator3(data_tree):
      assert(len(data_tree.axes) == 3)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2 in v1.keys():
               yield {data_tree.axes[0]: k0,
                      data_tree.axes[1]: k1,
                      data_tree.axes[2]: k2,}

   def MultiAxesTreeKeyGenerator4(data_tree):
      assert(len(data_tree.axes) == 4)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3 in v2.keys():
                  yield {data_tree.axes[0]: k0,
                         data_tree.axes[1]: k1,
                         data_tree.axes[2]: k2,
                         data_tree.axes[3]: k3,}

   def MultiAxesTreeKeyGenerator5(data_tree):
      assert(len(data_tree.axes) == 5)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  for k4 in v3.keys():
                     yield {data_tree.axes[0]: k0,
                            data_tree.axes[1]: k1,
                            data_tree.axes[2]: k2,
                            data_tree.axes[3]: k3,
                            data_tree.axes[4]: k4,}

   def MultiAxesTreeItemGenerator0(data_tree):
      assert(len(data_tree.axes) == 0)
      if False:
         yield ({},None,)

   def MultiAxesTreeItemGenerator1(data_tree):
      assert(len(data_tree.axes) == 1)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         yield ({data_tree.axes[0]: k0,}, v0,)

   def MultiAxesTreeItemGenerator2(data_tree):
      assert(len(data_tree.axes) == 2)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            yield ({data_tree.axes[0]: k0,
                    data_tree.axes[1]: k1,}, v1,)

   def MultiAxesTreeItemGenerator3(data_tree):
      assert(len(data_tree.axes) == 3)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               yield ({data_tree.axes[0]: k0,
                       data_tree.axes[1]: k1,
                       data_tree.axes[2]: k2,}, v2,)

   def MultiAxesTreeItemGenerator4(data_tree):
      assert(len(data_tree.axes) == 4)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  yield ({data_tree.axes[0]: k0,
                          data_tree.axes[1]: k1,
                          data_tree.axes[2]: k2,
                          data_tree.axes[3]: k3,}, v3,)

   def MultiAxesTreeItemGenerator5(data_tree):
      assert(len(data_tree.axes) == 5)
      assert(data_tree.data)
      for k0, v0 in data_tree.data.items():
         for k1, v1 in v0.items():
            for k2, v2 in v1.items():
               for k3, v3 in v2.items():
                  for k4, v4 in v3.items():
                     yield ({data_tree.axes[0]: k0,
                             data_tree.axes[1]: k1,
                             data_tree.axes[2]: k2,
                             data_tree.axes[3]: k3,
                             data_tree.axes[4]: k4,}, v4,)

   def MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                                                axes_index, leftover_axes_index,
                                                val, depth):
      if data_tree.axes[depth] in partial_axes_index:
         key = partial_axes_index[data_tree.axes[depth]]
         if key in val:
            val = val[key]
            axes_index[data_tree.axes[depth]] = key
            if depth+1 == len(data_tree.axes):
               yield (axes_index.copy(), leftover_axes_index.copy(), val,)
            else:
               gen = Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                  axes_index, leftover_axes_index, val, depth+1)
               for yld in gen:
                  yield yld
         else:
            # print(data_tree, partial_axes_index,
            #       axes_index, leftover_axes_index,
            #       key, val, depth)
            raise NameError("invalid index {} {}".format(Data.get_axes_index_str(axes_index), Data.get_axis_index_str(data_tree.axes[depth], key)))
      else:
         for key, val in val.items():
            axes_index[data_tree.axes[depth]] = key
            leftover_axes_index[data_tree.axes[depth]] = key
            if depth+1 == len(data_tree.axes):
               yield (axes_index.copy(), leftover_axes_index.copy(), val,)
            else:
               gen = Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index,
                  axes_index, leftover_axes_index, val, depth+1)
               for yld in gen:
                  yield yld

   def MultiAxesTreePartialItemGenerator0(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 0)
      if False:
         yield ({},None,)

   def MultiAxesTreePartialItemGenerator1(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 1)
      assert(data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)

   def MultiAxesTreePartialItemGenerator2(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 2)
      assert(data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)

   def MultiAxesTreePartialItemGenerator3(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 3)
      assert(data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)

   def MultiAxesTreePartialItemGenerator4(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 4)
      assert(data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)

   def MultiAxesTreePartialItemGenerator5(data_tree, partial_axes_index):
      assert(len(data_tree.axes) == 5)
      assert(data_tree.data)
      return Data.MultiAxesTreePartialItemGenerator_helper(data_tree, partial_axes_index, {}, {}, data_tree.data, 0)

   class MultiAxesTree:
      # axes is an array of axis_indices in the depth order they occur in the tree
      # indices is a dictionary of axis_indices to indices

      def __init__(self, axes):
         assert(axes)
         self.axes = axes
         self.data = {}

      def check(self, axes_index):
         data = self.data
         for axis_index in self.axes:
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(axis_name))
            index = axes_index[axis_index]
            if not index in data:
               return False
            data = data[index]
         return True

      def get(self, axes_index):
         data = self.data
         for axis_index in self.axes:
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(axis_name))
            index = axes_index[axis_index]
            if not index in data:
               raise NameError("Missing index {}".format(index))
            data = data[index]
         return data

      def set(self, axes_index, val):
         data = self.data
         for i in range(0, len(self.axes)-1):
            axis_index = self.axes[i]
            if not axis_index in axes_index:
               axis_name = Data.axes[axis_index]
               raise NameError("Missing axis {}".format(axis_name))
            index = axes_index[axis_index]
            if not index in data:
               data[index] = {}
            data = data[index]
         axis_index = self.axes[len(self.axes)-1]
         if not axis_index in axes_index:
            axis_name = Data.axes[axis_index]
            raise NameError("Missing axis {}".format(axis_name))
         index = axes_index[axis_index]
         data[index] = val

      def indexName(self, axes_index):
         name = ""
         for axis_index, index in axes_index.items():
            if name:
               name = "{} {}".format(name, Data.get_index_name(axis_index, index))
            else:
               name = Data.get_index_name(axis_index, index)
         return name

      def axesString(self):
         axes_names = ""
         for axis_index in self.axes:
            if axes_names:
               axes_names = "{}, {}".format(axes_names, Data.axes[axis_index])
            else:
               axes_names = "[{}".format(Data.axes[axis_index])
         return "{}]".format(axes_names)

      def dataString(self):
         buf = ""
         for axes_index, val in self.data.items():
            index_buf = " "
            for axis_index, index in axes_index.items():
               index_buf = "{} {}".format(index_buf, Data.get_index_name(axis_index, index))
            buf += "{} {}".format(buf, val)
         return buf

      def __repr__(self):
         return "MultiAxesTree({}):\n{}".format(self.axesString(), self.dataString())

      def __str__(self):
         return "MultiAxesTree({})".format(self.axesString())

      def keys(self):
         assert(self.data != None)
         if len(self.axes) == 0:
            return Data.MultiAxesTreeKeyGenerator0(self)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreeKeyGenerator1(self)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreeKeyGenerator2(self)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreeKeyGenerator3(self)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreeKeyGenerator4(self)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreeKeyGenerator5(self)
         else:
            raise ValueError

      def items(self):
         assert(self.data != None)
         if len(self.axes) == 0:
            return Data.MultiAxesTreeItemGenerator0(self)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreeItemGenerator1(self)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreeItemGenerator2(self)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreeItemGenerator3(self)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreeItemGenerator4(self)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreeItemGenerator5(self)
         else:
            raise ValueError

      def partial_match_items(self, partial_axes_index):
         assert(self.data != None)
         num_matching_indices = 0
         for axis_index in  self.axes:
            if axis_index in partial_axes_index:
               num_matching_indices += 1
         assert(num_matching_indices == len(partial_axes_index))
         if len(self.axes) == 0:
            return Data.MultiAxesTreePartialItemGenerator0(self, partial_axes_index)
         elif len(self.axes) == 1:
            return Data.MultiAxesTreePartialItemGenerator1(self, partial_axes_index)
         elif len(self.axes) == 2:
            return Data.MultiAxesTreePartialItemGenerator2(self, partial_axes_index)
         elif len(self.axes) == 3:
            return Data.MultiAxesTreePartialItemGenerator3(self, partial_axes_index)
         elif len(self.axes) == 4:
            return Data.MultiAxesTreePartialItemGenerator4(self, partial_axes_index)
         elif len(self.axes) == 5:
            return Data.MultiAxesTreePartialItemGenerator5(self, partial_axes_index)
         else:
            raise ValueError

      def __iter__(self):
         return self.keys()


   class DataTree:

      def __init__(self, kind, label, model_kind=None, axes=None, args=None, func=None):
         self.kind = kind
         self.label = label
         self.axes = axes
         self.args = args
         self.func = func
         self.model_kind = model_kind
         if not self.model_kind and self.args:
            self.model_kind = self.args[0]
         self.data = None

      def makeData(self, axes=None):
         if not self.axes:
            if axes:
               self.axes = axes
            elif self.model_kind and self.model_kind in Data.kinds:
               self.axes = Data.kinds[self.model_kind].axes
         assert(self.axes)
         self.data = Data.MultiAxesTree(self.axes)

      def hasAxes(self, other_axes):
         for axis_index in other_axes:
            if not axis_index in self.axes:
               return False
         return True

      def sameAxes(self, other_axes):
         if len(self.axes) != len(other_axes):
            return False
         self.hasAxes(other_axes)

      def missingAxes(self, other_axes):
         for axis_index in other_axes:
            if not axis_index in self.axes:
               return True
         return False

      def check(self, axes_index):
         return self.data.check(axes_index)

      def get(self, axes_index):
         return self.data.get(axes_index)

      def set(self, axes_index, val):
         return self.data.set(axes_index, val)

      def keys(self):
         return self.data.keys()

      def items(self):
         return self.data.items()

      def partial_match_items(self, partial_axes_index):
         return self.data.partial_match_items(partial_axes_index)

      def __iter__(self):
         return iter(self.data)

      def indexName(self, axes_index):
         return self.data.indexName(axes_index)

      def axesString(self):
         return self.data.axesString()

      def dataString(self):
         return self.data.dataString()

      def __repr__(self):
         return "DataTree({} {} {}):\n{}".format(self.kind, self.label, self.axesString(), self.dataString())

      def __str__(self):
         return "DataTree({} {} {})".format(self.kind, self.label, self.axesString())

   class DataTreeTemplate:

      def __init__(self, kind_template, label_template,
                   combined_axis=None, model_kind=None, args=None, func=None):
         self.kind_template = kind_template
         self.label_template = label_template
         self.combined_axis_template = combined_axis
         self.model_kind_template = model_kind
         self.arg_templates = args
         self.func = func

      def getKind(self, template_args):
         return self.kind_template.format(*template_args)

      def getLabel(self, template_args):
         arg_labels = [arg_kind in Data.kinds and Data.kinds[arg_kind].label or None for arg_kind in template_args]
         return self.label_template.format(*arg_labels)

      def getArgs(self, template_args):
         return [ arg.format(*template_args) for arg in self.arg_templates ]

      def getCombinedAxis(self, template_args):
         return self.combined_axis_template.format(*template_args)

      def getModelKind(self, args, template_args):
         assert(len(args) > 0)
         model_kind = None
         # choose model_kind with most axes
         for kind in args:
            if kind in Data.kinds:
               if not model_kind:
                  model_kind = kind
               elif len(Data.kinds[kind].axes) > len(Data.kinds[model_kind].axes):
                  model_kind = kind
         # use chosen model_kind
         if self.model_kind_template:
            model_kind = self.model_kind_template.format(*template_args)
         assert(model_kind)
         return model_kind

      def getAxes(self, model_kind, template_args):
         model_axes = Data.kinds[model_kind].axes
         combined_axis_index = None
         if self.combined_axis_template:
            combined_axis_name = self.getCombinedAxis(template_args)
            combined_axis_index = Data.axes[combined_axis_name]
         axes = []
         for axis_index in model_axes:
            if axis_index != combined_axis_index:
               axes.append(axis_index)
         return axes

      def makeDataTree(self, template_args):
         kind = self.getKind(template_args)
         label = self.getLabel(template_args)
         args = self.getArgs(template_args)
         model_kind = self.getModelKind(args, template_args)
         axes = self.getAxes(model_kind, template_args)
         return Data.DataTree(kind, label, model_kind=model_kind, axes=axes, args=args, func=self.func)

   # has info derivable from first kind "time(s)" which is read from files
   kinds = { "Problem size":   DataTree("Problem size",   "Problem size", axes=info_axes),
             "Reps":           DataTree("Reps",           "Reps",         axes=info_axes),
             "Iterations/rep": DataTree("Iterations/rep", "Iterations",   axes=info_axes),
             "Kernels/rep":    DataTree("Kernels/rep",    "Kernels",      axes=info_axes),
             "Bytes/rep":      DataTree("Bytes/rep",      "Bytes",        axes=info_axes),
             "FLOPS/rep":      DataTree("FLOPS/rep",      "FLOPS",        axes=info_axes),

             "time(s)": DataTree("time(s)", "time(s)", axes=data_axes),

             "time(ms)": DataTree("time(ms)", "time(ms)", args=["time(s)"], func=lambda t: t * 1000.0),
             "time(us)": DataTree("time(us)", "time(us)", args=["time(s)"], func=lambda t: t * 1000000.0),
             "time(ns)": DataTree("time(ns)", "time(ns)", args=["time(s)"], func=lambda t: t * 1000000000.0),

             "time/rep(s)": DataTree("time/rep(s)", "time(s)", args=["time(s)", "Reps"], func=lambda t, r: t / r),
             "time/rep(ms)": DataTree("time/rep(ms)", "time(ms)", args=["time/rep(s)"], func=lambda tpr: tpr * 1000.0),
             "time/rep(us)": DataTree("time/rep(us)", "time(us)", args=["time/rep(s)"], func=lambda tpr: tpr * 1000000.0),
             "time/rep(ns)": DataTree("time/rep(ns)", "time(ns)", args=["time/rep(s)"], func=lambda tpr: tpr * 1000000000.0),

             "time/it(s)": DataTree("time/it(s)", "time(s)", args=["time/rep(s)", "Iterations/rep"], func=lambda tpr, ipr: tpr / ipr),
             "time/it(ms)": DataTree("time/it(ms)", "time(ms)", args=["time/it(s)"], func=lambda tpi: tpi * 1000.0),
             "time/it(us)": DataTree("time/it(us)", "time(us)", args=["time/it(s)"], func=lambda tpi: tpi * 1000000.0),
             "time/it(ns)": DataTree("time/it(ns)", "time(ns)", args=["time/it(s)"], func=lambda tpi: tpi * 1000000000.0),

             "time/kernel(s)": DataTree("time/kernel(s)", "time(s)", args=["time/rep(s)", "Kernels/rep"], func=lambda tpr, kpr: tpr / kpr),
             "time/kernel(ms)": DataTree("time/kernel(ms)", "time(ms)", args=["time/kernel(s)"], func=lambda tpk: tpk * 1000.0),
             "time/kernel(us)": DataTree("time/kernel(us)", "time(us)", args=["time/kernel(s)"], func=lambda tpk: tpk * 1000000.0),
             "time/kernel(ns)": DataTree("time/kernel(ns)", "time(ns)", args=["time/kernel(s)"], func=lambda tpk: tpk * 1000000000.0),

             "throughput(Problem size/s)": DataTree("throughput(Problem size/s)", "throughput(Problem size/s)", args=["time/rep(s)", "Problem size"], func=lambda tpr, ps: ps / tpr),
             "throughput(Problem size/ms)": DataTree("throughput(Problem size/ms)", "throughput(Problem size/ms)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000.0),
             "throughput(Problem size/us)": DataTree("throughput(Problem size/us)", "throughput(Problem size/us)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000.0),
             "throughput(Problem size/ns)": DataTree("throughput(Problem size/ns)", "throughput(Problem size/ns)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000000.0),
             "throughput(KProblem size/s)": DataTree("throughput(KProblem size/s)", "throughput(KProblem size/s)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000.0),
             "throughput(MProblem size/s)": DataTree("throughput(MProblem size/s)", "throughput(MProblem size/s)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000.0),
             "throughput(GProblem size/s)": DataTree("throughput(GProblem size/s)", "throughput(GProblem size/s)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000000.0),
             "throughput(TProblem size/s)": DataTree("throughput(TProblem size/s)", "throughput(TProblem size/s)", args=["throughput(Problem size/s)"], func=lambda thr: thr / 1000000000000.0),

             "bandwidth(B/s)": DataTree("bandwidth(B/s)", "bandwidth(B/s)", args=["time/rep(s)", "Bytes/rep"], func=lambda tpr, bpr: bpr / tpr),
             "bandwidth(KB/s)": DataTree("bandwidth(KB/s)", "bandwidth(KB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1000.0),
             "bandwidth(MB/s)": DataTree("bandwidth(MB/s)", "bandwidth(MB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1000000.0),
             "bandwidth(GB/s)": DataTree("bandwidth(GB/s)", "bandwidth(GB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1000000000.0),
             "bandwidth(TB/s)": DataTree("bandwidth(TB/s)", "bandwidth(TB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1000000000000.0),
             "bandwidth(KiB/s)": DataTree("bandwidth(KiB/s)", "bandwidth(KiB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1024.0),
             "bandwidth(MiB/s)": DataTree("bandwidth(MiB/s)", "bandwidth(MiB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1048576.0),
             "bandwidth(GiB/s)": DataTree("bandwidth(GiB/s)", "bandwidth(GiB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1073741824.0),
             "bandwidth(TiB/s)": DataTree("bandwidth(TiB/s)", "bandwidth(TiB/s)", args=["bandwidth(B/s)"], func=lambda bps: bps / 1099511627776.0),

             "FLOPS": DataTree("FLOPS", "FLOPS", args=["time/rep(s)", "FLOPS/rep"], func=lambda tpr, fpr: fpr / tpr),
             "KFLOPS": DataTree("KFLOPS", "KFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000.0),
             "MFLOPS": DataTree("MFLOPS", "MFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000000.0),
             "GFLOPS": DataTree("GFLOPS", "GFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000000000.0),
             "TFLOPS": DataTree("TFLOPS", "TFLOPS", args=["FLOPS"], func=lambda fps: fps / 1000000000000.0),

      }

   kind_templates = {
             "log10": DataTreeTemplate("log10<{0}>", "log10({0})", args=["{0}",], func=lambda val: math.log(val, 10)),
             "log2": DataTreeTemplate("log2<{0}>", "log2({0})", args=["{0}",], func=lambda val: math.log(val, 2)),
             "ln": DataTreeTemplate("ln<{0}>", "ln({0})", args=["{0}",], func=lambda val: math.log(val)),

             "add": DataTreeTemplate("add<{0},{1}>", "{0} + {1}", args=["{0}", "{1}"], func=lambda lhs, rhs: lhs + rhs),
             "sub": DataTreeTemplate("sub<{0},{1}>", "{0} - {1}", args=["{0}", "{1}"], func=lambda lhs, rhs: lhs - rhs),
             "mul": DataTreeTemplate("mul<{0},{1}>", "{0} * {1}", args=["{0}", "{1}"], func=lambda lhs, rhs: lhs * rhs),
             "div": DataTreeTemplate("div<{0},{1}>", "{0} / {1}", args=["{0}", "{1}"], func=lambda lhs, rhs: lhs / rhs),

             "first": DataTreeTemplate("first<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=first),
             "last": DataTreeTemplate("last<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=last),
             "min": DataTreeTemplate("min<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=min),
             "max": DataTreeTemplate("max<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=max),
             "sum": DataTreeTemplate("sum<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=sum),
             "avg": DataTreeTemplate("avg<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=avg),
             "stddev": DataTreeTemplate("stddev<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=stddev),
             "relstddev": DataTreeTemplate("relstddev<{0},{1}>", "{0}", combined_axis="{1}", args=["{0}"], func=relstddev),

             "_LR": DataTreeTemplate("_LR<{0}>", "intercept, slope, correlation coefficient", combined_axis="run_size", args=["{0}", "Problem size"], func=linearRegression),
             "LR_intercept": DataTreeTemplate("LR_intercept<{0}>", "intercept", args=["_LR<{0}>"], func=lambda lr: lr[0]),
             "LR_slope": DataTreeTemplate("LR_slope<{0}>", "slope", args=["_LR<{0}>"], func=lambda lr: lr[1]),
             "LR_correlationCoefficient": DataTreeTemplate("LR_correlationCoefficient<{0}>", "correlation coefficient", args=["_LR<{0}>"], func=lambda lr: lr[2]),
             "LR": DataTreeTemplate("LR<{0}>", "{0}", model_kind="{0}", args=["_LR<{0}>", "Problem size"], func=eval_linearRegression),

             "_LR_log": DataTreeTemplate("_LR_log<{0}>", "intercept, slope, correlation coefficient", combined_axis="run_size", args=["{0}", "Problem size"], func=linearRegression_loglog),
             "LR_log_intercept": DataTreeTemplate("LR_log_intercept<{0}>", "intercept", args=["_LR_log<{0}>"], func=lambda lr: lr[0]),
             "LR_log_slope": DataTreeTemplate("LR_log_slope<{0}>", "slope", args=["_LR_log<{0}>"], func=lambda lr: lr[1]),
             "LR_log_correlationCoefficient": DataTreeTemplate("LR_log_correlationCoefficient<{0}>", "correlation coefficient", args=["_LR_log<{0}>"], func=lambda lr: lr[2]),
             "LR_log": DataTreeTemplate("LR_log<{0}>", "{0}", model_kind="{0}", args=["_LR_log<{0}>", "Problem size"], func=eval_linearRegression_loglog),

             "_LR2": DataTreeTemplate("_LR2<{0}>", "intercept, slope, correlation coefficient", combined_axis="run_size", args=["{0}", "Problem size"], func=segmented_linearRegression),
             "LR2_intercept": DataTreeTemplate("LR2_intercept<{0}>", "intercept", args=["_LR2<{0}>"], func=lambda lr: lr[0]),
             "LR2_slope": DataTreeTemplate("LR2_slope<{0}>", "slope", args=["_LR2<{0}>"], func=lambda lr: lr[1]),
             "LR2_correlationCoefficient": DataTreeTemplate("LR2_correlationCoefficient<{0}>", "correlation coefficient", args=["_LR2<{0}>"], func=lambda lr: lr[2]),
             "LR2": DataTreeTemplate("LR2<{0}>", "{0}", model_kind="{0}", args=["_LR2<{0}>", "Problem size"], func=eval_segmented_linearRegression),

             "_LR2_log": DataTreeTemplate("_LR2_log<{0}>", "intercept, slope, correlation coefficient", combined_axis="run_size", args=["{0}", "Problem size"], func=segmented_linearRegression_loglog),
             "LR2_log_intercept": DataTreeTemplate("LR2_log_intercept<{0}>", "intercept", args=["_LR2_log<{0}>"], func=lambda lr: lr[0]),
             "LR2_log_slope": DataTreeTemplate("LR2_log_slope<{0}>", "slope", args=["_LR2_log<{0}>"], func=lambda lr: lr[1]),
             "LR2_log_correlationCoefficient": DataTreeTemplate("LR2_log_correlationCoefficient<{0}>", "correlation coefficient", args=["_LR2_log<{0}>"], func=lambda lr: lr[2]),
             "LR2_log": DataTreeTemplate("LR2_log<{0}>", "{0}", model_kind="{0}", args=["_LR2_log<{0}>", "Problem size"], func=eval_segmented_linearRegression_loglog),

      }

   def compute_data(kind):
      # print("compute_data", kind)
      if not kind in Data.kinds:
         raise NameError("Unknown data kind {}".format(kind))

      datatree = Data.kinds[kind]
      if datatree.data:
         return # already calculated

      if not (datatree.model_kind and datatree.args and datatree.func):
         raise NameError("Computing data is not supported for kind {0}".format(kind))

      model_kind = datatree.model_kind
      compute_args = datatree.args
      compute_func = datatree.func

      if model_kind != kind:
         Data.compute(model_kind)

      arg_datatrees = ()
      for arg_kind in compute_args:
         # calculate data for arg_kind
         Data.compute(arg_kind)
         arg_datatree = Data.kinds[arg_kind]
         arg_datatrees = arg_datatrees + (arg_datatree,)

      if (not model_kind in Data.kinds) or (not Data.kinds[model_kind].data):
         raise NameError("Model data not available {0}, no args".format(model_kind))

      datatree.makeData()

      use_lists = ()
      for arg_datatree in arg_datatrees:
         use_list = datatree.missingAxes(arg_datatree.axes)
         use_lists = use_lists + (use_list,)

      for axes_index in Data.kinds[model_kind]:

         if not datatree.check(axes_index):
            args_val = ()
            for i in range(0, len(arg_datatrees)):
               arg_datatree = arg_datatrees[i]
               arg_val = arg_datatree.get(axes_index)
               if use_lists[i]:
                  arg_val = [arg_val,]
               args_val = args_val + (arg_val,)
            datatree.set(axes_index, args_val)
         else:
            args_val = datatree.get(axes_index)
            for i in range(0, len(arg_datatrees)):
               if use_lists[i]:
                  arg_datatree = arg_datatrees[i]
                  arg_val = arg_datatree.get(axes_index)
                  args_val[i].append(arg_val)

      for axes_index, args_val in datatree.items():
         val = compute_func(*args_val)
         datatree.set(axes_index, val)

   def compute_index(kind_preindex, index_args):
      # print("compute_index", kind_preindex, index_args)
      Data.compute(kind_preindex)
      datatree_preindex = Data.kinds[kind_preindex]

      # extract axes and indices
      partial_axis_index = {}
      for index_str in index_args:
         index_list = index_str.split("::")
         if len(index_list) != 2:
            raise NameError("Expected valid index <axis>::<index>: {}".format(index_str))
         axis_name = index_list[0].strip()
         index_name = index_list[1].strip()
         partial_axis_index.update(Data.get_axis_index(axis_name, index_name))

      kind = "{}[{}]".format(kind_preindex, ",".join(index_args))

      datatree = None
      if kind in Data.kinds:
         datatree = Data.kinds[kind]
         if datatree.data:
            return
      else:
         axes = Data.axes_difference(datatree_preindex.axes, partial_axis_index)
         datatree = Data.DataTree(kind, datatree_preindex.label, axes=axes)
         Data.kinds[kind] = datatree

      datatree.makeData()

      for axes_index, partial_axes_index, value in datatree_preindex.partial_match_items(partial_axis_index):
         datatree.set(partial_axes_index, value)

   def compute_templated_data(kind_template, template_args):
      # print("compute_templated_data", kind_template, template_args)
      if kind_template in Data.kind_templates:
         kind = Data.kind_templates[kind_template].getKind(template_args)
         if not kind in Data.kinds:
            # compute args first to ensure arg kinds exist
            for arg_kind in Data.kind_templates[kind_template].getArgs(template_args):
               Data.compute(arg_kind)
            Data.kinds[kind] = Data.kind_templates[kind_template].makeDataTree(template_args)
            Data.compute(kind)
      else:
         raise NameError("Unkown kind template {}".format(kind_template))

   def kind_template_scan(kind):
      # print("kind_template_scan", kind)

      kind_prefix = None

      template_args = []
      index_args = []

      template_depth = 0
      index_depth = 0

      arg_end_idx = -1

      # look through string backwards to find indexing or templating
      for i_forward in range(0, len(kind)):
         i = len(kind) - i_forward - 1
         c = kind[i]
         if c == ">" or c == "]":
            if template_depth == 0 and index_depth == 0:
               arg_end_idx = i
            if c == ">":
               template_depth += 1
            elif c == "]":
               index_depth += 1
         elif c == ",":
            if template_depth == 1 and index_depth == 0:
               template_args.append(kind[i+1:arg_end_idx].strip())
               arg_end_idx = i
            elif template_depth == 0 and index_depth == 1:
               index_args.append(kind[i+1:arg_end_idx].strip())
               arg_end_idx = i
         elif c == "<" or c == "[":
            if template_depth == 1 and index_depth == 0:
               template_args.append(kind[i+1:arg_end_idx].strip())
               arg_end_idx = -1
            elif template_depth == 0 and index_depth == 1:
               index_args.append(kind[i+1:arg_end_idx].strip())
               arg_end_idx = -1
            if c == "<":
               template_depth -= 1
            elif c == "[":
               index_depth -= 1
            if template_depth == 0 and index_depth == 0:
               if not kind_prefix:
                  kind_prefix = kind[:i].strip()
                  break
      assert(arg_end_idx == -1)
      assert(template_depth == 0)
      assert(index_depth == 0)
      assert(kind_prefix)

      # reverse lists
      for i in range(0, len(template_args)//2):
         i_rev = len(template_args) - i - 1
         template_args[i], template_args[i_rev] = template_args[i_rev], template_args[i]
      for i in range(0, len(index_args)//2):
         i_rev = len(index_args) - i - 1
         index_args[i], index_args[i_rev] = index_args[i_rev], index_args[i]

      return (kind_prefix, template_args, index_args)

   def compute(kind):
      if kind in Data.kinds:
         if not Data.kinds[kind].data:
            Data.compute_data(kind)
         else:
            pass
      else:
         kind_template, template_args, index_args = Data.kind_template_scan(kind)
         # print("Data.kind_template_scan", kind_template, template_args, index_args)
         if template_args:
            if kind_template in Data.kind_templates:
               Data.compute_templated_data(kind_template, template_args)
            else:
               raise NameError("Unknown data kind template {}".format(kind))
         elif index_args:
            Data.compute_index(kind_template, index_args)
         else:
            raise NameError("Unknown data kind {}".format(kind))



def get_size_from_dir_name(sweep_subdir_name):
   # print(sweep_subdir_name)
   run_size_name = sweep_subdir_name.replace("SIZE_", "")
   try:
      run_size = int(run_size_name)
      return str(run_size)
   except ValueError:
      raise NameError("Expected SIZE_<run_size>".format(sweep_subdir_name))

def read_runinfo_file(sweep_index, sweep_subdir_runinfo_file_path, run_size_index):
   # print(sweep_index, sweep_subdir_runinfo_file_path, run_size_index)
   with open(sweep_subdir_runinfo_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      ignore = True
      c_to_info_kinds = {}
      for row in file_reader:
         # print(row)
         if row[0].strip() == "Kernels":
            ignore = False
            for c in range(1, len(row)):
               info_kind = row[c].strip()
               # print(c, info_kind)
               if not info_kind in Data.kinds:
                  # add new kind to global data
                  print("Unknown kernel info {0}".format(info_kind))
                  Data.kinds[info_kind] = Data.DataTree(info_kind, "info", Data.info_axes)
               if info_kind in c_to_info_kinds:
                  print("Repeated kernel info {0}".format(info_kind))
                  sys.exit(1)
               if not Data.kinds[info_kind].data:
                  # add data to kind
                  Data.kinds[info_kind].makeData()
               if not sweep_index in Data.kinds[info_kind].data.data:
                  # add new sweep to global data
                  Data.kinds[info_kind].data.data[sweep_index] = {}
               if run_size_index in Data.kinds[info_kind].data.data[sweep_index]:
                  sweep_dir_name = Data.get_index_name(Data.axes["sweep_dir_name"], sweep_index)
                  run_size_name = Data.get_index_name(Data.axes["run_size"], run_size_index)
                  print("Repeated kernel size {0} in {1}".format(sweep_dir_name, run_size_name))
                  sys.exit(1)
               else:
                  # add new size to global data
                  Data.kinds[info_kind].data.data[sweep_index][run_size_index] = {}
               # make map of columns to names
               c_to_info_kinds[c] = info_kind
               c_to_info_kinds[info_kind] = c
         elif not ignore:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in Data.kernels:
               kernel_index = Data.kernels[kernel_name]
            elif (len(Data.include_kernels) == 0 or kernel_name in Data.include_kernels) and (not kernel_name in Data.exclude_kernels):
               # add kernel to global list
               Data.add_kernel(kernel_name)
               kernel_index = Data.kernels[kernel_name]
            else:
               continue # skip this kernel

            for c in range(1, len(row)):
               info_kind = c_to_info_kinds[c]
               try:
                  # add data to global structure
                  val = int(row[c].strip())
                  # print(kernel_index, kernel_name, info_kind, val)

                  axes_index = { Data.axes["sweep_dir_name"]: sweep_index,
                                 Data.axes["run_size"]: run_size_index,
                                 Data.axes["kernel_index"]: kernel_index, }

                  Data.kinds[info_kind].set(axes_index, val)
               except ValueError:
                  pass # could not convert data to int


def read_timing_file(sweep_index, sweep_subdir_timing_file_path, run_size_index):
   # print(sweep_index, sweep_subdir_timing_file_path, run_size_index)
   with open(sweep_subdir_timing_file_path, "r") as file:
      file_reader = csv.reader(file, delimiter=',')

      data_kind = g_timing_file_kind
      if not data_kind in Data.kinds:
         raise NameError("Unknown kind {}".format(data_kind))
      if not Data.kinds[data_kind].data:
         Data.kinds[data_kind].makeData()
      if not sweep_index in Data.kinds[data_kind].data.data:
         Data.kinds[data_kind].data.data[sweep_index] = {}
      if not run_size_index in Data.kinds[data_kind].data.data[sweep_index]:
         Data.kinds[data_kind].data.data[sweep_index][run_size_index] = {}
      else:
         sweep_dir_name = Data.get_index_name(Data.axes["sweep_dir_name"], sweep_index)
         run_size_name = Data.get_index_name(Data.axes["run_size"], run_size_index)
         raise NameError("Already seen {0} in {1}".format(sweep_dir_name, run_size_name))

      c_to_variant_index = {}
      c_to_tuning_index = {}
      for row in file_reader:
         # print(row)
         if row[0].strip() == "Kernel":
            if len(c_to_variant_index) == 0:
               for c in range(1, len(row)):
                  variant_name = row[c].strip()
                  variant_index = -1
                  if variant_name in Data.variants:
                     variant_index = Data.variants[variant_name]
                  elif (len(Data.include_variants) == 0 or variant_name in Data.include_variants) and (not variant_name in Data.exclude_variants):
                     Data.add_variant(variant_name)
                     variant_index = Data.variants[variant_name]
                  else:
                     variant_index = -1
                  c_to_variant_index[c] = variant_index

            elif len(c_to_tuning_index) == 0:
               for c in range(1, len(row)):
                  tuning_name = row[c].strip()
                  tuning_index = None
                  if tuning_name in Data.tunings:
                     tuning_index = Data.tunings[tuning_name]
                  elif (len(Data.include_tunings) == 0 or tuning_name in Data.include_tunings) and (not tuning_name in Data.exclude_tunings):
                     Data.add_tuning(tuning_name)
                     tuning_index = Data.tunings[tuning_name]
                  else:
                     tuning_index = -1
                  c_to_tuning_index[c] = tuning_index

            else:
               print("Unknown row {0}".format(row))
               sys.exit(1);
         elif len(c_to_variant_index) > 0 and len(c_to_tuning_index) > 0:
            kernel_index = -1
            kernel_name = row[0].strip()
            if kernel_name in Data.kernels:
               kernel_index = Data.kernels[kernel_name]
            else:
               continue # skip kernel

            for c in range(1, len(row)):
               variant_index = c_to_variant_index[c]
               tuning_index = c_to_tuning_index[c]
               if variant_index < 0 or tuning_index < 0:
                  continue # ignore data

               axes_index = { Data.axes["sweep_dir_name"]: sweep_index,
                              Data.axes["run_size"]: run_size_index,
                              Data.axes["kernel_index"]: kernel_index,
                              Data.axes["variant_index"]: variant_index,
                              Data.axes["tuning_index"]: tuning_index, }

               try:
                  val = float(row[c].strip())
                  # print(kernel_index, kernel_name, variant_index, tuning_index, data_kind, val)
                  Data.kinds[data_kind].set(axes_index, val)
               except ValueError:
                  pass # could not convert data to float


def get_plot_data(kind, partial_axes_index):

   if not kind in Data.kinds:
      raise NameError("Unknown kind {}".format(kind))

   kind_data = Data.kinds[kind]

   assert(kind_data.hasAxes(partial_axes_index))

   data = []
   for axes_index, leftover_axes_index, value in kind_data.partial_match_items(partial_axes_index):
      index_name = kind_data.indexName(leftover_axes_index)
      data.append({ "name": index_name,
                    "axes_index": leftover_axes_index,
                    "data": [value] })

   return data

def get_plot_data2(xkind, ykind, partial_axes_index):

   if not ykind in Data.kinds:
      raise NameError("Unknown kind {}".format(ykind))
   if not xkind in Data.kinds:
      raise NameError("Unknown kind {}".format(xkind))

   ykind_data = Data.kinds[ykind]
   xkind_data = Data.kinds[xkind]

   assert(ykind_data.hasAxes(partial_axes_index))
   assert(xkind_data.hasAxes(partial_axes_index))

   data = []
   for axes_index, leftover_axes_index, yvalue in ykind_data.partial_match_items(partial_axes_index):
      index_name = ykind_data.indexName(leftover_axes_index)
      xvalue = xkind_data.get(axes_index)
      data.append({ "name": index_name,
                    "axes_index": leftover_axes_index,
                    "ydata": [yvalue],
                    "xdata": [xvalue] })

   return data

g_gname = None

g_ylabel = None
g_yscale = None
g_ylim = None

g_xlabel = None
g_xscale = None
g_xlim = None

g_hbin_size = None

def plot_data_split_line(outputfile_name, split_axis_name, xaxis_name, xkind, ykinds):
   # print("plotting {} {} {} {}".format(outputfile_name, split_axis_name, xaxis_name, xkind, ykinds))

   assert(split_axis_name == "kernel_index")
   for split_index in range(0, Data.num_kernels):
      split_name = Data.kernels[split_index]

      ylabel = g_ylabel
      yscale = g_yscale or "log"
      ylim = g_ylim

      xlabel = g_xlabel or Data.kinds[xkind].label
      xscale = g_xscale or "log"
      xlim = g_xlim

      gname = g_gname

      split_data = { "ynames": [],
                     "ycolor": {},
                     "yformat": {},
                     "ydata": {},
                     "xdata": {}, }

      for ykind in ykinds:
         if gname:
            gname = "{}\n{}".format(gname, ykind)
         else:
            gname = "{}".format(ykind)
         if not ykind in Data.kinds:
            raise NameError("Unknown kind {}".format(ykind))
         if not ylabel:
            ylabel = Data.kinds[ykind].label
         elif (not g_ylabel) and ylabel != Data.kinds[ykind].label:
            raise NameError("kinds use different labels {}".format([Data.kinds[_ykind].label for _ykind in ykinds]))

         assert(xaxis_name == "run_size")
         for x_index in range(0, Data.num_run_sizes):

            axes_index = { Data.axes[split_axis_name]: split_index,
                           Data.axes[xaxis_name]: x_index  }

            data_list = get_plot_data2(xkind, ykind, axes_index)

            for data in data_list:
               yname = data["name"]

               if not yname in split_data["ydata"]:

                  ycolor = (0.0, 0.0, 0.0, 1.0)
                  if Data.axes["variant_index"] in data["axes_index"]:
                     variant_index = data["axes_index"][Data.axes["variant_index"]]
                     ycolor = Data.variant_colors[variant_index]

                  ymarker = ""
                  if Data.axes["sweep_dir_name"] in data["axes_index"]:
                     sweep_index = data["axes_index"][Data.axes["sweep_dir_name"]]
                     ymarker = Data.sweep_markers[sweep_index]

                  yformat = "{}-".format(ymarker)
                  if Data.axes["tuning_index"] in data["axes_index"]:
                     tuning_index = data["axes_index"][Data.axes["tuning_index"]]
                     yformat = "{}{}".format(ymarker, Data.tuning_formats[tuning_index])

                  split_data["ynames"].append(yname)
                  split_data["ycolor"][yname] = ycolor
                  split_data["yformat"][yname] = yformat
                  split_data["ydata"][yname] = []
                  split_data["xdata"][yname] = []

               split_data["ydata"][yname].append(data["ydata"][0])
               split_data["xdata"][yname].append(data["xdata"][0])

      fname = "{}_{}.png".format(outputfile_name, split_name)
      if gname:
         gname = "{}\n{}".format(split_name, gname)
      else:
         gname = "{}".format(split_name)

      print("Plotting {}:".format(fname))

      for yname in split_data["ynames"]:

         ycolor = split_data["ycolor"][yname]
         yformat = split_data["yformat"][yname]
         ydata = split_data["ydata"][yname]
         xdata = split_data["xdata"][yname]

         if yname in g_series_reformat and "format" in g_series_reformat[yname]:
            yformat = g_series_reformat[yname]["format"]
         if yname in g_series_reformat and "color" in g_series_reformat[yname]:
            ycolor = g_series_reformat[yname]["color"]

         print("  series \"{}\" format \"{}\" color \"{}\"".format(yname, yformat, ycolor))

         if len(ykinds) > 1:
            yname = "{} {}".format(Data.kinds[ykind].kind, yname)

         plt.plot(xdata,ydata,yformat,color=ycolor,label=yname)

      if ylabel:
         plt.ylabel(ylabel)
      if yscale:
         plt.yscale(yscale)
      if ylim:
         plt.ylim(ylim)

      if xlabel:
         plt.xlabel(xlabel)
      if xscale:
         plt.xscale(xscale)
      if xlim:
         plt.xlim(xlim)

      plt.title(gname)
      plt.legend()
      plt.grid(True)

      plt.savefig(fname, dpi=150.0)
      plt.clf()

def plot_data_bar(outputfile_name, xaxis, ykinds):
   # print("plotting {} {} {}".format(outputfile_name, xaxis, ykinds))

   assert(xaxis == "kernel_index")

   gname = g_gname

   xlabel = g_xlabel or "Kernel"
   xscale = g_xscale
   xlim = g_xlim

   ylabel = g_ylabel
   yscale = g_yscale
   ylim = g_ylim

   for ykind in ykinds:
      if gname:
         gname = "{}\n{}".format(gname, ykind)
      else:
         gname = "{}".format(ykind)
      if not ykind in Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not ylabel:
         ylabel = Data.kinds[ykind].label
      elif (not g_ylabel) and ylabel != Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([Data.kinds[_ykind].label for _ykind in ykinds]))

   kernel_data = { "kernel_names": [],
                   "kernel_centers": [],
                   "ynames": {},
                   "ycolor": {},
                   "ydata": {}, }

   for kernel_index in range(0, Data.num_kernels):
      kernel_name = Data.kernels[kernel_index]

      kernel_data["kernel_names"].append(kernel_name)
      kernel_data["kernel_centers"].append(kernel_index)

      axes_index = { Data.axes["kernel_index"]: kernel_index }

      for ykind in ykinds:

         ydata_list = get_plot_data(ykind, axes_index)

         for ydata in ydata_list:

            assert(len(ydata["data"]) == 1)

            yname = ydata["name"]
            if len(ykinds) > 1:
               yname = "{} {}".format(Data.kinds[ykind].kind, yname)

            ycolor = (0.0, 0.0, 0.0, 1.0)
            if Data.axes["variant_index"] in ydata["axes_index"]:
               variant_index = ydata["axes_index"][Data.axes["variant_index"]]
               ycolor = Data.variant_colors[variant_index]

            if not yname in kernel_data["ynames"]:
               kernel_data["ynames"][yname] = len(kernel_data["ynames"])
               kernel_data["ycolor"][yname] = ycolor
               kernel_data["ydata"][yname] = []

            # pad with 0s if find missing data
            while len(kernel_data["ydata"][yname])+1 < len(kernel_data["kernel_names"]):
               kernel_data["ydata"][yname].append(0.0)

            kernel_data["ydata"][yname].append(ydata["data"][0])

   fname = "{}.png".format(outputfile_name)
   if not gname:
      gname = "{}".format("bar")

   print("Plotting {}:".format(fname))

   num_xticks = len(kernel_data["kernel_centers"])
   plt.figure(figsize=(max(num_xticks*0.5, 4), 6,))

   y_n = len(kernel_data["ydata"])
   ywidth = 1.0 / (y_n+1)
   for yname in kernel_data["ynames"]:

      y_i = kernel_data["ynames"][yname]
      ycolor = kernel_data["ycolor"][yname]
      yaxis = kernel_data["ydata"][yname]

      if yname in g_series_reformat and "color" in g_series_reformat[yname]:
         ycolor = g_series_reformat[yname]["color"]

      print("  series \"{}\" color \"{}\"".format(yname, ycolor))

      xaxis = [c + (y_i+1)/(y_n+1) - 0.5 for c in kernel_data["kernel_centers"]]

      # pad with 0s if find missing data
      while len(yaxis) < len(kernel_data["kernel_names"]):
         yaxis.append(0.0)

      plt.bar(xaxis,yaxis,label=yname,width=ywidth,color=ycolor,zorder=3) # ,edgecolor="grey")

   xticks = kernel_data["kernel_centers"]
   xtick_names = kernel_data["kernel_names"]

   if ylabel:
      plt.ylabel(ylabel)
   if yscale:
      plt.yscale(yscale)
   if ylim:
      plt.ylim(ylim)

   if xlabel:
      plt.xlabel(xlabel)
   if xscale:
      plt.xscale(xscale)
   if xlim:
      plt.xlim(xlim)

   plt.xticks(xticks, xtick_names, rotation=90)

   plt.title(gname)
   plt.legend()
   plt.grid(True, zorder=0)

   plt.savefig(fname, dpi=150.0, bbox_inches="tight")
   plt.clf()


def plot_data_histogram(outputfile_name, haxis, hkinds):
   # print("plotting {} {} {}".format(outputfile_name, haxis, hkinds))

   assert(haxis == "kernel_index")

   gname = g_gname

   hbin_size = g_hbin_size
   hbin_max = None
   hbin_min = None

   xlabel = g_xlabel
   xscale = g_xscale
   xlim = g_xlim

   ylabel = g_ylabel or "Number"
   yscale = g_yscale
   ylim = g_ylim

   for ykind in hkinds:
      if gname:
         gname = "{}\n{}".format(gname, ykind)
      else:
         gname = "{}".format(ykind)
      if not ykind in Data.kinds:
         raise NameError("Unknown kind {}".format(ykind))
      if not xlabel:
         xlabel = Data.kinds[ykind].label
      elif (not g_xlabel) and xlabel != Data.kinds[ykind].label:
         raise NameError("kinds use different labels {}".format([Data.kinds[_ykind].label for _ykind in hkinds]))

   if not hbin_size:

      hdata_all = []

      for kernel_index in range(0, Data.num_kernels):
         kernel_name = Data.kernels[kernel_index]

         axes_index = { Data.axes["kernel_index"]: kernel_index }

         for ykind in hkinds:

            hdata_list = get_plot_data(ykind, axes_index)

            for hdata in hdata_list:

               assert(len(hdata["data"]) == 1)
               hdata_all.append(hdata["data"][0])

      hdata_all.sort()

      num_hdata = len(hdata_all)
      i_Q1 = math.floor(num_hdata * 0.25)
      i_Q3 = math.floor(num_hdata * 0.75)
      hdata_Q1 = hdata_all[i_Q1]
      hdata_Q3 = hdata_all[i_Q3]
      iqr = hdata_Q3 - hdata_Q1

      hbin_size = 2.0 * iqr / num_hdata**(1.0/3.0)

      if hbin_size > 1.0:
         hbin_size = math.floor(hbin_size)
      elif hbin_size > 0.0:
         hbin_size = 1.0 / math.ceil(1.0 / hbin_size)
      else:
         hbin_size = 1.0

   kernel_data = { "hnames": {},
                   "hcolor": {},
                   "hbins": {}, }

   for kernel_index in range(0, Data.num_kernels):
      kernel_name = Data.kernels[kernel_index]

      axes_index = { Data.axes["kernel_index"]: kernel_index }

      for ykind in hkinds:

         hdata_list = get_plot_data(ykind, axes_index)

         for hdata in hdata_list:

            assert(len(hdata["data"]) == 1)

            hname = hdata["name"]
            if len(hkinds) > 1:
               hname = "{} {}".format(Data.kinds[ykind].kind, hname)

            hcolor = (0.0, 0.0, 0.0, 1.0)
            if Data.axes["variant_index"] in hdata["axes_index"]:
               variant_index = hdata["axes_index"][Data.axes["variant_index"]]
               hcolor = Data.variant_colors[variant_index]

            if not hname in kernel_data["hnames"]:
               kernel_data["hnames"][hname] = len(kernel_data["hnames"])
               kernel_data["hcolor"][hname] = hcolor
               kernel_data["hbins"][hname] = {}

            hbin = math.floor(hdata["data"][0] / hbin_size)

            if hbin_max == None or hbin > hbin_max:
               hbin_max = hbin
            if hbin_min == None or hbin < hbin_min:
               hbin_min = hbin

            if not hbin in kernel_data["hbins"][hname]:
               kernel_data["hbins"][hname][hbin] = 0
            kernel_data["hbins"][hname][hbin] += 1

   fname = "{}.png".format(outputfile_name)
   if not gname:
      gname = "{}".format("histogram")

   print("Plotting {}:".format(fname))

   num_xticks = hbin_max - hbin_min + 1
   if xlim:
      num_xticks = math.ceil((xlim[1] - xlim[0]) / hbin_size)
   plt.figure(figsize=(max(num_xticks*0.5, 4), 6,))

   h_n = len(kernel_data["hnames"])
   hwidth = hbin_size / h_n
   print(h_n, hwidth, hbin_size)
   for hname in kernel_data["hnames"]:

      h_i = kernel_data["hnames"][hname]
      xoffset = hbin_size * ((h_i+1)/(h_n+1) - 0.5)
      hcolor = kernel_data["hcolor"][hname]
      hbins = kernel_data["hbins"][hname]

      if hname in g_series_reformat and "color" in g_series_reformat[hname]:
         hcolor = g_series_reformat[hname]["color"]

      print("  series \"{}\" color \"{}\" offset {}".format(hname, hcolor, xoffset))

      xaxis = []
      haxis = []
      for i, hval in hbins.items():
         xval = (i + 0.5) * hbin_size + xoffset
         xaxis.append(xval)
         haxis.append(hval)

      plt.bar(xaxis,haxis,label=hname,width=hwidth,color=hcolor,zorder=3) # ,edgecolor="grey")

   if ylabel:
      plt.ylabel(ylabel)
   if yscale:
      plt.yscale(yscale)
   if ylim:
      plt.ylim(ylim)

   if xlabel:
      plt.xlabel(xlabel)
   if xscale:
      plt.xscale(xscale)
   if xlim:
      plt.xlim(xlim)

   plt.title(gname)
   plt.legend()
   plt.grid(True, zorder=0)

   plt.savefig(fname, dpi=150.0, bbox_inches="tight")
   plt.clf()


def main(argv):
   sweep_dir_paths = []
   outputfile = "graph"
   help_string = "sweep_graph.py -o <outputfile> <sweepdir1 [sweepdir2 ...]>"
   runinfo_filename = g_runinfo_filename
   timing_filename = g_timing_filename
   print_kinds = []
   split_line_graph_kind_lists = []
   bar_graph_kind_lists = []
   histogram_graph_kind_lists = []

   i = 0
   while i < len(argv):
      opt = argv[i]
      if len(opt) == 0:
         print(help_string)
         sys.exit(2)
      elif opt[0] == "-":

         handle_num = None
         handle_arg = None
         # no arg options
         if opt in ("-h", "--help"):
            print(help_string)
            sys.exit()
         # single arg options
         if opt in ("-o", "--output"):
            handle_num = 1
            def fo(arg):
               nonlocal outputfile
               outputfile = arg
            handle_arg = fo
         elif opt in ("-gname", "--graph-name"):
            handle_num = 1
            def gn(arg):
               global g_gname
               g_gname = arg
            handle_arg = gn
         elif opt in ("-ylabel", "--y-axis-label"):
            handle_num = 1
            def yl(arg):
               global g_ylabel
               g_ylabel = arg
            handle_arg = yl
         elif opt in ("-yscale", "--y-axis-scale"):
            handle_num = 1
            def ys(arg):
               global g_yscale
               g_yscale = arg
            handle_arg = ys
         elif opt in ("-xlabel", "--x-axis-label"):
            handle_num = 1
            def xl(arg):
               global g_xlabel
               g_xlabel = arg
            handle_arg = xl
         elif opt in ("-xscale", "--x-axis-scale"):
            handle_num = 1
            def xs(arg):
               global g_xscale
               g_xscale = arg
            handle_arg = xs
         elif opt in ("-hbin", "--histogram-bin-size"):
            handle_num = 1
            def hbin(arg):
               global g_hbin_size
               g_hbin_size = float(arg)
            handle_arg = hbin
         # two arg options
         elif opt in ("-ylim", "--y-axis-limit"):
            handle_num = 2
            def yl(ymin, ymax):
               global g_ylim
               g_ylim = (float(ymin), float(ymax))
            handle_arg = yl
         elif opt in ("-xlim", "--x-axis-limit"):
            handle_num = 2
            def xl(xmin, xmax):
               global g_xlim
               g_xlim = (float(xmin), float(xmax))
            handle_arg = xl
         elif opt in ("--recolor"):
            handle_num = 2
            def recolor(series_name, color_str):
               if not series_name in g_series_reformat:
                  g_series_reformat[series_name] = {}
               g_series_reformat[series_name]["color"] = make_color_tuple_str(color_str)
            handle_arg = recolor
         elif opt in ("--reformat"):
            handle_num = 2
            def reformat(series_name, format_str):
               if not series_name in g_series_reformat:
                  g_series_reformat[series_name] = {}
               g_series_reformat[series_name]["format"] = format_str
            handle_arg = reformat
         # multi arg options
         elif opt in ("-p", "--print"):
            handle_num = -1
            def p(arg):
               print_kinds.append(arg)
            handle_arg = p
         elif opt in ("-slg", "--split-line-graphs"):
            handle_num = -1
            split_line_graph_kind_lists.append([])
            def fslg(arg):
               split_line_graph_kind_lists[len(split_line_graph_kind_lists)-1].append(arg)
            handle_arg = fslg
         elif opt in ("-bg", "--bar-graph"):
            handle_num = -1
            bar_graph_kind_lists.append([])
            def fbg(arg):
               bar_graph_kind_lists[len(bar_graph_kind_lists)-1].append(arg)
            handle_arg = fbg
         elif opt in ("-hg", "--histogram-graph"):
            handle_num = -1
            histogram_graph_kind_lists.append([])
            def fhg(arg):
               histogram_graph_kind_lists[len(histogram_graph_kind_lists)-1].append(arg)
            handle_arg = fhg
         elif opt in ("-kg", "--kernel-groups"):
            handle_num = -1
            def fkg(arg):
               Data.include_kernel_groups[arg] = arg
            handle_arg = fkg
         elif opt in ("-ekg", "--exclude-kernel-groups"):
            handle_num = -1
            def fekg(arg):
               Data.exclude_kernel_groups[arg] = arg
            handle_arg = fekg
         elif opt in ("-k", "--kernels"):
            handle_num = -1
            def fk(arg):
               Data.include_kernels[arg] = arg
            handle_arg = fk
         elif opt in ("-ek", "--exclude-kernels"):
            handle_num = -1
            def fek(arg):
               Data.exclude_kernels[arg] = arg
            handle_arg = fek
         elif opt in ("-v", "--variants"):
            handle_num = -1
            def fv(arg):
               Data.include_variants[arg] = arg
            handle_arg = fv
         elif opt in ("-ev", "--exclude-variants"):
            handle_num = -1
            def fev(arg):
               Data.exclude_variants[arg] = arg
            handle_arg = fev
         elif opt in ("-t", "--tunings"):
            handle_num = -1
            def ft(arg):
               Data.include_tunings[arg] = arg
            handle_arg = ft
         elif opt in ("-et", "--exclude-tunings"):
            handle_num = -1
            def fet(arg):
               Data.exclude_tunings[arg] = arg
            handle_arg = fet
         elif opt in ("-es", "--exclude-sweeps"):
            handle_num = -1
            def fes(arg):
               Data.exclude_sweeps[arg] = arg
            handle_arg = fes

         # error unknown opt
         if handle_num == None:
            print(help_string)
            sys.exit(2)

         # fixed num args, handled together
         elif handle_num > 0:
            if not i+handle_num < len(argv):
               print("Missing option to {}".format(opt))
               sys.exit(2)
            args = []
            for h in range(1, handle_num+1):
               arg = argv[i+h]
               if arg[0] == "-":
                  print("Missing option to {}".format(opt))
                  sys.exit(2)
               args.append(arg)
            handle_arg(*args)
            i += handle_num

         # unfixed num args, handled one-by-one
         elif handle_num < 0:
            while i+1 < len(argv):
               arg = argv[i+1]
               if arg[0] == "-":
                  break
               handle_arg(arg)
               i += 1

      else:
         sweep_dir_paths.append(opt)
      i += 1

   for kernel_group in Data.include_kernel_groups.keys():
      if kernel_group in g_known_kernel_groups:
         for kernel_name in g_known_kernel_groups[kernel_group]["kernels"]:
            Data.include_kernels[kernel_name] = kernel_name
      else:
         print("Unknown kernel group {}".format(kernel_group))
         sys.exit(2)

   for kernel_group in Data.exclude_kernel_groups.keys():
      if kernel_group in g_known_kernel_groups:
         for kernel_name in g_known_kernel_groups[kernel_group]["kernels"]:
            Data.exclude_kernels[kernel_name] = kernel_name
      else:
         print("Unknown kernel group {}".format(kernel_group))
         sys.exit(2)

   print("Input directories are \"{0}\"".format(sweep_dir_paths))
   print("Output file is \"{0}\"".format(outputfile))

   for sweep_dir_path in sweep_dir_paths:
      sweep_dir_name = os.path.basename(sweep_dir_path)
      # print(sweep_dir_name, sweep_dir_path)

      if sweep_dir_name in Data.exclude_sweeps:
         continue

      if sweep_dir_name in Data.sweeps:
         raise NameError("Repeated sweep_dir_name {}".format(sweep_dir_name))
      Data.add_sweep(sweep_dir_name)
      sweep_index = Data.sweeps[sweep_dir_name]
      if sweep_index >= len(g_markers):
         raise NameError("Ran out of sweep markers for {}".format(sweep_dir_name))
      Data.sweep_markers[sweep_index] = g_markers[sweep_index]

      for r0,sweep_subdir_names,f0 in os.walk(sweep_dir_path):
         for sweep_subdir_name in sweep_subdir_names:
            sweep_subdir_path = os.path.join(sweep_dir_path, sweep_subdir_name)
            # print(sweep_dir_name, sweep_subdir_path)

            run_size_name = get_size_from_dir_name(sweep_subdir_name)

            if not run_size_name in Data.run_sizes:
               Data.add_run_size(run_size_name)
            run_size_index = Data.run_sizes[run_size_name]

            sweep_subdir_timing_file_path = ""
            sweep_subdir_runinfo_file_path = ""
            for r1,d1,sweep_subdir_file_names in os.walk(sweep_subdir_path):
               for sweep_subdir_file_name in sweep_subdir_file_names:
                  sweep_subdir_file_path = os.path.join(sweep_subdir_path, sweep_subdir_file_name)
                  if sweep_subdir_file_name == timing_filename:
                     sweep_subdir_timing_file_path = sweep_subdir_file_path
                  elif sweep_subdir_file_name == runinfo_filename:
                     sweep_subdir_runinfo_file_path = sweep_subdir_file_path

            if sweep_subdir_timing_file_path != "" and sweep_subdir_runinfo_file_path != "":
               # print(sweep_subdir_timing_file_path, sweep_subdir_runinfo_file_path)
               read_runinfo_file(sweep_index, sweep_subdir_runinfo_file_path, run_size_index)
               read_timing_file(sweep_index, sweep_subdir_timing_file_path, run_size_index)

   kinds_string = ""
   for kindTree in Data.kinds.values():
      kinds_string += ", {}".format(kindTree.kind)
   print("kinds")
   print("  {}".format(kinds_string[2:]))

   kind_templates_string = ""
   for kindTree_template in Data.kind_templates.values():
      kind_templates_string += ", {}".format(kindTree_template.kind_template)
   print("kind_templates")
   print("  {}".format(kind_templates_string[2:]))

   axes_string = ""
   for v in range(0, Data.num_axes):
      axes_string += ", {}".format(Data.axes[v])
   print("axes")
   print("  {}".format(axes_string[2:]))

   sweeps_string = ""
   for v in range(0, Data.num_sweeps):
      sweeps_string += ", {}".format(Data.sweeps[v])
   print("sweeps")
   print("  {}".format(sweeps_string[2:]))

   run_sizes_string = ""
   for v in range(0, Data.num_run_sizes):
      run_sizes_string += ", {}".format(Data.run_sizes[v])
   print("run_sizes")
   print("  {}".format(run_sizes_string[2:]))

   kernel_groups_string = ""
   for kernel_group in g_known_kernel_groups:
      kernel_groups_string += ", {}".format(kernel_group)
   print("kernel groups")
   print("  {}".format(kernel_groups_string[2:]))

   kernel_string = ""
   for v in range(0, Data.num_kernels):
      kernel_string += ", {}".format(Data.kernels[v])
   print("kernels")
   print("  {}".format(kernel_string[2:]))

   variant_string = ""
   for v in range(0, Data.num_variants):
      variant_string += ", {}".format(Data.variants[v])
   print("variants")
   print("  {}".format(variant_string[2:]))

   tuning_string = ""
   for v in range(0, Data.num_tunings):
      tuning_string += ", {}".format(Data.tunings[v])
   print("tunings")
   print("  {}".format(tuning_string[2:]))

   for kind in print_kinds:
      print("Print Data {}:".format(kind))
      Data.compute(kind)
      print(Data.kinds[kind].dataString())

   for kind_list in split_line_graph_kind_lists:
      print("Plot split line graph {}:".format(kind_list))
      for kind in kind_list:
         Data.compute(kind)
      plot_data_split_line(outputfile, "kernel_index", "run_size", "Problem size", kind_list)

   for kind_list in bar_graph_kind_lists:
      print("Plot bar graph {}:".format(kind_list))
      for kind in kind_list:
         Data.compute(kind)
      plot_data_bar(outputfile, "kernel_index", kind_list)

   for kind_list in histogram_graph_kind_lists:
      print("Plot histogram graph {}:".format(kind_list))
      for kind in kind_list:
         Data.compute(kind)
      plot_data_histogram(outputfile, "kernel_index", kind_list)

if __name__ == "__main__":
   main(sys.argv[1:])
